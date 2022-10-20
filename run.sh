for i in bikesharing compactiv cpusmall ctscan indoorloc mv pole puma32h telemonitoirng
do
    for j in 1 2 3
    do
        for m in 'input_perturbation' 'gradient_norm' 'mc_dropout' 'knowledge_distillation' 'ensemble'
        do
            for k in 0 1 2 3 4 5 6 7 8 9
            do
                python main.py $i $j $m $k
            done
        done
    done
done
