python example_jax.py
for i in 0 1 2 3
do
python symbolic.py --index=${i}
done
python plot_loss.py
python plot_params.py
python plot_c.py
