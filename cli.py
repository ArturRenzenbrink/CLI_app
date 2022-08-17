import argparse
import pickle

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-f', '--file', help='load lyrics via txt file')
group.add_argument('-l', '--line', help='load lyrics via line')
args = parser.parse_args()

with open('naive_classifier.bin', 'rb') as trained_model_file:
	model = pickle.load(trained_model_file)

if args.file:
	with open(args.file, 'r') as lyrics_file:
		lyrics = [lyrics_file.read()]

if args.line:
	lyrics = [args.line]

print(model.predict_proba(lyrics))
