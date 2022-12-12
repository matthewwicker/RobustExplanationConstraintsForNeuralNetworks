for eps in 0.0 0.05 0.10 0.15 0.2 0.25
do
    for alp in 0.0 0.05 0.10 0.15 0.2 0.25
    do
        python3 HalfmoonsTrain.py --alpha $alp --eps $eps
    done
done