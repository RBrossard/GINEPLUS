for i in 0 1 2 3 4
do
  python train_classifier.py --name gineplus -V -L 5 -H 400 -K 3 --conv-type gin+  --dataset molpcba -b 100 -l 0.001
done
python stats.py runs/molpcba/gineplus