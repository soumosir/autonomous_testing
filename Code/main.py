import argparse

import predict_turn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='1', type=int,  help='problem number you want to solve, options- 1/2/3,'
                                                                  ' Default: 1')
    parser.add_argument('--inDir', default='../Data/adaptive_hist_data',
                        help='path to input data folder, Default: ../Data/adaptive_hist_data')
    parser.add_argument('--outDir', default='../Data/output',
                        help='path to output folder where results will be stored, Default: ../Data/output')
    parser.add_argument('--video', default='../Data/challenge.mp4',
                        help='Video File Path, Default: ../Data/whiteline.mp4')
    parser.add_argument('--display', default=1, type=int,  help='display Results, Default: 1')
    parser.add_argument('--save', default=0, type=int, help='save Results, Default: 0')

    args = parser.parse_args()
    problem = int(args.problem)
    in_dir = args.inDir
    out_dir = args.outDir
    video_file = args.video
    display = bool(args.display)
    save = bool(args.save)

    if problem == 1:
        predict_turn.run(video_file, out_dir, display, save)
    else:
        print("[ERROR]: Wrong choice my friend. Choose again from 1, 2, 3")
        exit()
