# Matthew Wicker
for epsilon in 0.01 0.025 0.05 0.1
do
	for gamma in 0.0 #0.01 0.025 0.05 0.1
	do
		r=$(( $RANDOM % 5 ))
		CUDA_VISIBLE_DEVICES=$r, python3 mnisttrain.py --alp 1.0 --eps $epsilon --gam $gamma &
		r=$(( $RANDOM % 5 ))
		CUDA_VISIBLE_DEVICES=$r, python3 mnisttrain.py --alp 0.1 --eps $epsilon --gam $gamma &
		r=$(( $RANDOM % 5 ))
		CUDA_VISIBLE_DEVICES=$r, python3 mnisttrain.py --alp 0.25 --eps $epsilon --gam $gamma &
		r=$(( $RANDOM % 5 ))
		CUDA_VISIBLE_DEVICES=$r, python3 mnisttrain.py --alp 0.5 --eps $epsilon --gam $gamma &
		r=$(( $RANDOM % 5 ))
		CUDA_VISIBLE_DEVICES=$r, python3 mnisttrain.py --alp 0.75 --eps $epsilon --gam $gamma &
		wait
	done
done
