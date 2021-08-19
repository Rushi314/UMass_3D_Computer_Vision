import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import shutil
import glob
import argparse
import numpy as np
import random
from scipy.spatial.transform import Rotation

import torch
import torch.backends.cudnn as cudnn

from model import CorrNet
from model import PointNet
from utils import AverageMeter, mkdir_p, isdir, isfile

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# calculates matching accuracy
# vtx_feature is an output per-vertex feature vector from the input mesh (Nx32), where N is the number of vertices
# pts_feature is an output per-point feature vector from a point cloud(Mx32), where M is the number of points
def calc_matching_accuracy(vtx_feature, pts_feature, corr_gt, pts, thr):
    # feature_similarity is a NxM matrix storing the (cosine) similarity between mesh vertices and points    
    feature_similarity = torch.mm(vtx_feature, pts_feature.transpose(0, 1))
    # based on this matrix, we can compute for each vertex, the most similar point id (nnidx) and coordinates (nnpts)
    max_sim, nnidx = torch.max(feature_similarity, dim=1)
    nnpts = pts[nnidx]
    # note here that corr_gt stores ground-truth corresponding pairs of (mesh_vertex_id, point_id)
    # nnpts[corr_gt[i, 0]] returns the coordinates of the point that was found most similar to vertex i 
    # pts[corr_gt[i, 1]] returns the coordinates of the point that is the actual ground-truth corresondence to vertex i
    # then the Euclidean distance of all the above pairs of two points is computed 
    dist = torch.sqrt(torch.sum((nnpts[corr_gt[:, 0]] - pts[corr_gt[:, 1]]) ** 2, dim=-1))
    # if the distance is smaller than this threshold, we consider it correct match
    acc = torch.sum((dist <= thr).float()) / len(dist)
    return acc


# calculates the correspondence mask Nx1
# each mesh vertex i might have a correspondence with a point in the point cloud, or not 
# the reason is that the point cloud is partial, and might not have some parts 
# (e.g., the back of a scanned human)
# the predicted mask is Nx1 probabilities, which must be thresholded 
# to compare with the ground-truth mask, which is binary
def calc_mask_accuracy(pred_mask, gt_mask):
    gt_mask = gt_mask >= 0.5
    acc = torch.sum((pred_mask >= 0.5) == gt_mask).float() / len(pred_mask)
    return acc


# function to save a checkpoint during training, including the best model so far
# as explained later: you will train the model to learn mesh vertex features and point cloud features (first training stage)
# then you train the model to learn the correspondence mask (second training stage)
def save_checkpoint(state, is_best, train_corrmask, checkpoint_folder='checkpoints/', filename='checkpoint.pth.tar'):
    # second stage: save checkpoints for correspondence mask training
    if train_corrmask:
        checkpoint_file = os.path.join(checkpoint_folder, 'checkpoint_corr_{}.pth.tar'.format(state['epoch']))
        torch.save(state, checkpoint_file)
        if is_best:
            shutil.copyfile(checkpoint_file, os.path.join(checkpoint_folder, 'model_corr_best.pth.tar'))
    # first stage: save checkpoints for feature learning
    else:
        checkpoint_file = os.path.join(checkpoint_folder, 'checkpoint_{}.pth.tar'.format(state['epoch']))
        torch.save(state, checkpoint_file)
        if is_best:
            shutil.copyfile(checkpoint_file, os.path.join(checkpoint_folder, 'model_best.pth.tar'))


# training function, read carefully
def train(train_pts_filelist, model, optimizer, args):
    model.train()  # switch to train mode
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    random.shuffle(train_pts_filelist)
    num_batch = np.floor(len(train_pts_filelist) / args.train_batch).astype(int)
    # print(len(train_pts_filelist))
    for i in range(num_batch):
        # print(i)
        pts_filelist_batch = train_pts_filelist[i * args.train_batch: (i + 1) * args.train_batch]
        # print("here2")
        optimizer.zero_grad()
        for pts_filename in pts_filelist_batch:
            frame_num = int(pts_filename.split("/")[-1].split("_")[-2])
            # load data
            pts = np.load(pts_filename)
            vtx = np.load(pts_filename.replace("_rpts.npy", "_vtx.npy"))
            corr = np.load(pts_filename.replace("_rpts.npy", "_corr.npy"))
            mask = np.load(pts_filename.replace("_rpts.npy", "_corrmask.npy"))
            # convert to tensor
            pts_tensor = torch.FloatTensor(pts).to(device)
            vtx_tensor = torch.FloatTensor(vtx).to(device)
            corr_tensor = torch.LongTensor(corr).to(device)
            mask_tensor = torch.FloatTensor(mask).to(device)
            # forward pass
            vtx_feature, pts_feature, pred_mask = model(vtx_tensor, pts_tensor)
            # calculate loss and accuracy
            if args.train_corrmask:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_mask.squeeze(dim=1), mask_tensor)
                acc = calc_mask_accuracy(pred_mask.squeeze(dim=1), mask_tensor)
            else:
                loss = NTcrossentropy(vtx_feature, pts_feature, corr_tensor, tau=args.tau_nce)
                acc = calc_matching_accuracy(vtx_feature, pts_feature, corr_tensor, pts_tensor, args.distance_threshold)
            loss.backward()
            loss_meter.update(loss.item())
            acc_meter.update(acc.item())
        # we accumulate gradient for several samples, and make a step by the average gradient
        optimizer.step()
    return loss_meter.avg, acc_meter.avg


# testing function, read carefully
def test(test_pts_filelist, model, args, save_results):
    model.eval()  # switch to test mode
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    rotations = []
    random.shuffle(test_pts_filelist)
    num_batch = np.floor(len(test_pts_filelist) / args.test_batch).astype(int)
    for i in range(num_batch):
        pts_filelist_batch = test_pts_filelist[i * args.test_batch: (i + 1) * args.test_batch]
        for pts_filename in pts_filelist_batch:
            frame_num = int(pts_filename.split("/")[-1].split("_")[-2])
            # load data
            pts = np.load(pts_filename)
            vtx = np.load(pts_filename.replace("_rpts.npy", "_vtx.npy"))
            corr = np.load(pts_filename.replace("_rpts.npy", "_corr.npy"))
            mask = np.load(pts_filename.replace("_rpts.npy", "_corrmask.npy"))
            # convert to tensor
            pts_tensor = torch.FloatTensor(pts).to(device)
            vtx_tensor = torch.FloatTensor(vtx).to(device)
            corr_tensor = torch.LongTensor(corr).to(device)
            mask_tensor = torch.FloatTensor(mask).to(device)
            # forward pass
            with torch.no_grad():
                vtx_feature, pts_feature, pred_mask = model(vtx_tensor, pts_tensor)
            # calculate loss and accuracy
            if args.train_corrmask:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_mask.squeeze(dim=1), mask_tensor)
                pred_mask = torch.sigmoid(pred_mask)
                acc = calc_mask_accuracy(pred_mask.squeeze(dim=1), mask_tensor)
                rotations.append(
                    fit_rotation(vtx_tensor, pts_tensor, vtx_feature, pts_feature, pred_mask.squeeze(dim=1)))
            else:
                loss = NTcrossentropy(vtx_feature, pts_feature, corr_tensor, tau=args.tau_nce)
                acc = calc_matching_accuracy(vtx_feature, pts_feature, corr_tensor, pts_tensor, args.distance_threshold)
            loss_meter.update(loss.item())
            acc_meter.update(acc)
            # we may save the features for further debugging / visualizations
            if save_results:
                mkdir_p("output/")
                np.save(f"output/{frame_num}_vtx.npy", vtx)
                np.save(f"output/{frame_num}_rpts.npy", pts)
                np.save(f"output/{frame_num}_vtx_feature.npy", vtx_feature.cpu().data.numpy())
                np.save(f"output/{frame_num}_rpts_feature.npy", pts_feature.cpu().data.numpy())
                if args.train_corrmask:
                    np.save(f"output/{frame_num}_pred_corrmask.npy", pred_mask.cpu().data.numpy())
    if args.train_corrmask:
        # average over all the recovered rotations
        rotations = np.mean(rotations, axis=0)
        # convert quaternions back to Euler angles to be more intuitive.
        print("Average fitted rotation: ", Rotation.from_quat(rotations).as_euler('xyz', degrees=True))
    return loss_meter.avg, acc_meter.avg


# given: a Nx32 per-vertex feature vector for the mesh,
#        a Mx32 per-point feature vector in the input pount cloud,
#        a corr matrix Kx2 that stores K ground-truth pairs of corresponding vertices-points
# the function should output the normalized temperature-scaled cross entropy loss
# **** YOU SHOULD CHANGE THIS FUNCTION, CURRENTLY IT IS INCORRECT ****
def NTcrossentropy(vtx_feature, pts_feature, corr, tau=0.07):
    # cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    # corr_vtx_feature = vtx_feature[corr[:, 0]]
    # prod = torch.randn(corr_vtx_feature.size(0), pts_feature.size(0), requires_grad=True).to(device)
    # label = corr[:, 1]
    # loss = cross_entropy_loss(prod, label)

    similarity = torch.exp(torch.matmul(vtx_feature, torch.transpose(pts_feature, 0, 1)) / tau)

    loss = (-1) * torch.sum(
        torch.log(similarity[corr[:, 0], corr[:, 1]]) - torch.log(torch.sum(similarity[corr[:, 0], :], dim=1)))

    return loss


# function to estimate a rotation matrix to align the vertices and the points based on the predicted reliable correspondences.
# **** YOU SHOULD CHANGE THIS FUNCTION ****
def fit_rotation(vtx, pts, vtx_feature, pts_feature, corrmask):
    sim_matrix = torch.matmul(vtx_feature, pts_feature.T)[corrmask>0.5]
    pt_s, idx = torch.max(sim_matrix, dim=1)
    sim_pts = pts[idx]
    vtx_pts = vtx[corrmask > 0.5]
    centroid_p = torch.mean(vtx_pts,0)
    centroid_q = torch.mean(sim_pts,0)
    P = vtx_pts - centroid_p
    Q = sim_pts - centroid_q
    scale = torch.pow(torch.norm(P, p=2)/torch.norm(Q, p=2), 1/4)
    centroid_p_new = centroid_p/scale
    centroid_q_new = centroid_q/scale
    P = vtx_pts - centroid_p_new
    Q = sim_pts - centroid_q_new
    S = torch.matmul(torch.transpose(P,0,1),Q)
    U,D,V = torch.svd(S)
    R = torch.matmul(U,torch.transpose(V,0,1))
    R = Rotation.from_matrix(R)
    return R.as_quat()
    # keep the following line, transform the estimated to rotation matrix to a quaternion
    # the starter code handles the rest


# main function, read carefully to understand the code
def main(args):
    best_acc = 0.0

    # create checkpoint folder
    if not isdir(args.checkpoint_folder):
        print("Creating new checkpoint folder " + args.checkpoint_folder)
        mkdir_p(args.checkpoint_folder)

    # create model
    model = CorrNet(num_output_features=args.num_output_features, train_corrmask=args.train_corrmask)
    model.to(device)
    print("=> Will use the (" + device.type + ") device.")

    # cudnn will optimize execution for our network   
    cudnn.benchmark = True

    # optionally resume training from a file inside the checkpoint folder
    if args.resume_file:
        path_to_resume_file = os.path.join(args.checkpoint_folder, args.resume_file)
        if isfile(path_to_resume_file):
            print("=> Loading training checkpoint '{}'".format(path_to_resume_file))
            checkpoint = torch.load(path_to_resume_file)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> No previous checkpoint found at '{}'".format(path_to_resume_file))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)
    print("=> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    train_pts_filelist = glob.glob(os.path.join(args.train_folder, "*_rpts.npy"))
    val_pts_filelist = glob.glob(os.path.join(args.val_folder, "*_rpts.npy"))
    test_pts_filelist = glob.glob(os.path.join(args.test_folder, "*_rpts.npy"))

    # will do testing only, print out test loss/accuracy and finish (assuming a trained model)
    if args.evaluate:
        print("\nEvaluation only")
        test_loss, test_acc = test(test_pts_filelist, model, args, save_results=True)
        print(f" test_loss: {test_loss:.8f}. test_acc: {test_acc:.8f}.")
        return

    # Training mode
    # Train first the feature extractor
    # then train the correspondence mask using this argument --train_corrmask
    # the correspondence mask training freezes the (pretrained) feature extractor
    if args.train_corrmask:
        print("=> Will now train the correspondence mask")
        for name, param in model.pointnet_share.named_parameters():
            param.requires_grad = False
    else:
        print("=> Will now train the feature extractor")

    # perform training!
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=args.gamma)
    for epoch in range(args.start_epoch, args.epochs):
        lr = scheduler.get_last_lr()
        print(f"\nEpoch: {epoch + 1:d} | LR: {lr[0]:.8f}")
        train_loss, train_acc = train(train_pts_filelist, model, optimizer, args)
        val_loss, val_acc = test(val_pts_filelist, model, args, save_results=False)
        if args.save_results and (epoch == args.epochs - 1):
            test_loss, test_acc = test(test_pts_filelist, model, args, save_results=True)
        else:
            test_loss, test_acc = test(test_pts_filelist, model, args, save_results=False)
        scheduler.step()
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            print("=> Current model is the best according to validation accuracy")
        save_checkpoint({"epoch": epoch + 1, "state_dict": model.state_dict(), "best_acc": best_acc,
                         "optimizer": optimizer.state_dict()},
                        is_best, args.train_corrmask, checkpoint_folder=args.checkpoint_folder)
        print(f"Epoch{epoch + 1:d}. train_loss: {train_loss:.8f}. train_acc: {train_acc:.8f}.")
        print(f"Epoch{epoch + 1:d}. val_loss: {val_loss:.8f}. val_acc: {val_acc:.8f}.")
        print(f"Epoch{epoch + 1:d}. test_loss: {test_loss:.8f}. test_acc: {test_acc:.8f}.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='depth mesh corresponce')
    parser.add_argument("-e", "--evaluate", action="store_false", #If command line parameter is given then stores true else default false
                        help="Activate test mode - Evaluate model on val/test set (no training)")

    # paths you may want to adjust, but it's better to keep the defaults
    parser.add_argument("--checkpoint_folder", default="checkpoints/", type=str, help="Folder to save checkpoints")
    parser.add_argument("--resume_file", default="model_best.pth.tar", type=str,
                        help="Path to retrieve latest checkpoint file relative to checkpoint folder")
    parser.add_argument("--train_folder", default="dataset/train/", type=str,
                        help="Folder where training data are stored")
    parser.add_argument("--val_folder", default="dataset/val/", type=str,
                        help="Folder where validation data are stored")
    parser.add_argument("--test_folder", default="dataset/test/", type=str, help="Folder where test data are stored")

    # hyperameters of network/options for training
    parser.add_argument("--num_output_features", default=32, type=str, help="chn number of output feature")
    parser.add_argument("--start_epoch", default=0, type=int, help="Start from specified epoch number")
    parser.add_argument("--schedule", type=int, nargs="+", default=[40],
                        help="Decrease learning rate at these milestone epochs.")
    parser.add_argument("--gamma", default=0.1, type=float,
                        help="Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestone epochs")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Number of epochs to train (when loading a previous model, it will train for an extra number of epochs)")
    parser.add_argument("--weight_decay", default=1e-3, type=float, help="Weight decay/L2 regularization on weights")
    parser.add_argument("--lr", default=1e-2, type=float, help="Initial learning rate")
    parser.add_argument("--train_batch", default=8, type=int, help="Batch size for training")
    parser.add_argument("--train_corrmask", action="store_false", help="Train also the correspondence mask branch")
    parser.add_argument("--tau_nce", default=0.07, type=float,
                        help="Parameter used in the temperature-scaled cross entropy loss")

    # various options for testing and evaluation
    parser.add_argument("--save_results", default=True, action="store_true",
                        help="Save results during testing - useful for visualization")
    parser.add_argument("--test_batch", default=8, type=int,
                        help="Batch size for testing (used for faster evaluation since it parallelizes inference for testing data)")
    parser.add_argument("--distance_threshold", default=0.04, type=float,
                        help="for each mesh vertex, the method computes its most similar point in the point cloud. \
                              That point might not be correct. The actual corresponding point has a distance d from it \
                              We consider a correspondence correct if d <= distance_threshold")

    print(parser.parse_args())
    main(parser.parse_args())
