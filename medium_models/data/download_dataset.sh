wget https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar
tar xvf datasets.tar

echo "*** Use GLUE-SST-2 as default SST-2 ***"
mv original/SST-2 original/SST-2-original
mv original/GLUE-SST-2 original/SST-2

echo "*** Done ***"
