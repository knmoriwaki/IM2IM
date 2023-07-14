import sys
import os
import argparse
import time
import numpy as np
import pdb

from explain.xai_dataloader import XAIDataLoader
from explain.xai_calc import calc_importance, compare_exp_testset
from explain.xai_plot import plot_r_occ_sample, plot_perturbed_map, plot_occlusion_sensitivity, plot_all_r_vs_k, plot_r_single_sample, plot_true_fake_maps

parser = argparse.ArgumentParser(description="")
parser.add_argument("--isRef", dest="isRef", action='store_true', help="Reference run, no XAI experiments")
parser.add_argument('--ref_name', dest="ref_name", default='test', help='Any name for the reference run')
parser.add_argument("--xai_exp", dest="xai_exp", type=str, default=None, 
                    help="Evaluation of respective XAI experiment.")

parser.add_argument('--output_dir', dest="output_dir", type=str, default='./output', help='all the outputs and models are saved here')
parser.add_argument('--results_dir', dest="results_dir", type=str, default='./output/xai_results', help='optional, inference outputs are saved here')
parser.add_argument("--nrun", dest="nrun", type=int, default=100, help="number of realizations")
parser.add_argument("--nindex", dest="nindex", type=int, default=1, help="number of indexes for each realization")
parser.add_argument("--occlusion_size", type=int, default=64, help="Occlusion window size for occlusion sensitivity")


args = parser.parse_args()

def main():
    start_time = time.time()
    # Check if output directories exists
    if args.isRef and not os.path.exists(os.path.join(args.output_dir, args.ref_name)):
        print(os.path.join(args.output_dir, args.ref_name), " Reference run does not exist")
        sys.exit(0)
    if args.xai_exp and not os.path.exists(os.path.join(args.output_dir, args.xai_exp)):
        print(os.path.join(args.output_dir, args.xai_exp), " XAI experiment does not exist")
        sys.exit(0)
    # Check if results directories exists, if not create them
    if args.isRef and not os.path.exists(os.path.join(args.results_dir, args.ref_name)):
        os.makedirs(os.path.join(args.results_dir, args.ref_name))
        print("Created directory: ", os.path.join(args.results_dir, args.ref_name))
    if args.xai_exp and not os.path.exists(os.path.join(args.results_dir, args.xai_exp)):
        os.makedirs(os.path.join(args.results_dir, args.xai_exp))
        print("Created directory: ", os.path.join(args.results_dir, args.xai_exp))

    prefix_list = [ "run{:d}_index{:d}".format(i, j) for i in range(args.nrun) for j in range(args.nindex) ]
    # pick a random prefix
    prefix = np.random.choice(prefix_list)
    if args.isRef:
        print("Reference run, no XAI experiment.")
        # Until now I rather compared the expierments to the reference run rather than evaluating it itself. 
        # As an idea I could make a blue plot but with r between true and fake maps instead between ref and exp? 
        sys.exit(0)
    else:
        print("Evaluating XAI experiment: ", args.xai_exp)
        nbins=10
        log_bins=True
        results_dir = os.path.join(args.results_dir, args.xai_exp)
        if args.xai_exp != 'xai_exp_occlusion':
            r_mix_list, r_ha_list, r_oiii_list = compare_exp_testset(args.output_dir, args.ref_name, args.xai_exp, 
                                                                    nrun=args.nrun, nindex=args.nindex, nbins=nbins, log_bins=log_bins)
            plot_all_r_vs_k(r_mix_list, results_dir, 
                            title="Comparison of reconstructed mixed signals")
            plot_all_r_vs_k(r_ha_list, results_dir, 
                            title="Comparison of reconstructed Ha signals")
            plot_all_r_vs_k(r_oiii_list, results_dir, 
                            title="Comparison of reconstructed OIII signals")
            data = XAIDataLoader(args.output_dir, args.xai_exp, suffix=prefix)
            plot_perturbed_map(data, results_dir, args.xai_exp, suffix=prefix)
            plot_r_single_sample(data, results_dir, suffix=prefix, nbins=nbins, log_bins=log_bins)
        else:
            total_n_occ = int(256*256/(args.occlusion_size**2))
            plot_r_occ_sample(args.output_dir, args.ref_name, args.xai_exp, results_dir, total_n_occ, 
                              suffix=prefix, nbins=nbins, log_bins=log_bins)
            im_mix, im_ha, im_oiii = calc_importance(args.output_dir, args.ref_name, args.xai_exp, 
                                                     total_n_occ, suffix=prefix, nbins=nbins, log_bins=log_bins)
            plot_occlusion_sensitivity(im_mix, im_ha, im_oiii, results_dir)
            for i in range(total_n_occ):
                df = XAIDataLoader(args.output_dir, args.xai_exp, suffix=prefix, n_occ=i)
                plot_perturbed_map(df, results_dir, args.xai_exp, n_occ=i, suffix=prefix)



    print("Time for evaluation: {:.2f} sec".format(time.time() - start_time))


if __name__ == "__main__":
    main()