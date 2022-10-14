for i in ctscan
do
    for j in 1 2 3
    do
        for m in input_perturbation gradient_norm mc_dropout datafree_kd ensemble
        do
            for k in 0 1 2 3 4 5 6 7 8 9
            do
                python main.py $i $j $m $k
            done
        done
    done
done
