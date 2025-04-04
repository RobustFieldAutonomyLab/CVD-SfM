from pathlib import Path
from SfM.feature_extraction import run_disk_extraction
from SfM.feature_matching import generate_and_match_pairs
from SfM.reconstruction import run_reconstruction

images = Path('')
outputs = Path('')

features = outputs / 'features.h5'
matches = outputs / 'matches.h5'
pairs = outputs / 'pairs.txt'
sfm_output_dir = outputs / 'sfm_output'
prior_csv = './cross_view/cross_estimation.csv'

run_disk_extraction(images, features)
generate_and_match_pairs(pairs, features, matches)
run_reconstruction(sfm_output_dir, images, pairs, features, matches, pose_prior_csv=prior_csv)