echo "Interpolation Inductive Biases"
for freq in 4
do
    # echo "Frequency: ${freq} with Transformer no Curriculum"
    # python fourier_series_inductive_bias.py ${freq} interpolate Transformer False DFT False
    # echo "Frequency: ${freq} with Transformer with Curriculum"
    # python fourier_series_inductive_bias.py ${freq} interpolate Transformer True DFT False
    # echo "Frequency: ${freq} with Fourier LSQ"
    # python fourier_series_inductive_bias.py ${freq} interpolate "Fourier LSQ" False DFT False


    # echo "Frequency: ${freq} with Transformer no Curriculum With Pure Frequencies"
    # python fourier_series_inductive_bias.py ${freq} interpolate Transformer False DFT True
    # echo "Frequency: ${freq} with Transformer with Curriculum With Pure Frequencies"
    # python fourier_series_inductive_bias.py ${freq} interpolate Transformer True DFT True
    # echo "Frequency: ${freq} with Fourier LSQ With Pure Frequencies"
    # python fourier_series_inductive_bias.py ${freq} interpolate "Fourier LSQ" False DFT True
    
    # echo "Frequency: ${freq} with BMA"
    # python fourier_series_mixture_inductive_bias.py ${freq} interpolate BMA DFT False

    echo "Frequency: ${freq} with Transformer"
    python fourier_series_mixture_inductive_bias.py ${freq} interpolate Transformer DFT False
    # echo "Frequency: ${freq} with DFT"
    # python fourier_series_inductive_bias.py ${freq} interpolate DFT
done



# echo "Extrapolation Inductive Biases"
# for freq in 1 2 3 4 5 6 7 8 9 10
# do
#     echo "Frequency: ${freq}"
#     python fourier_series_inductive_bias.py ${freq} extrapolate
# done