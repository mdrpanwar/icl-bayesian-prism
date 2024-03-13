

for freq in 2 3 4 5 6 7 8 9 10 11 12 13 14
do
    # echo "Running for Frequency: ${freq} for Transformer"
    # python fourier_series_eval.py ${freq} interpolate "Transformer" True 42 False
    # echo "Running for Frequency: ${freq} for Transformer"
    # python fourier_series_eval.py ${freq} interpolate "Transformer" False 42 False
    # echo "Running for Frequency: ${freq} for LSQ"
    # python fourier_series_eval.py ${freq} interpolate "Fourier LSQ" False 42 False

    echo "Running for Frequency: ${freq} for Transformer With Pure Frequencies"
    python fourier_series_eval.py ${freq} interpolate "Transformer" True 42 True
    echo "Running for Frequency: ${freq} for Transformer With Pure Frequencies"
    python fourier_series_eval.py ${freq} interpolate "Transformer" False 42 True
    echo "Running for Frequency: ${freq} for LSQ With Pure Frequencies"
    python fourier_series_eval.py ${freq} interpolate "Fourier LSQ" False 42 True
done