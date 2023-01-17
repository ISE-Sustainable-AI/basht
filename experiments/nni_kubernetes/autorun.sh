umask 0000

for nworkers in 1 2 4; do
	for cpus in 1  2  3 4; do
		for grid in small medium large; do
			if [[ $nworkers == 1 && $cpus -ne 2 ]]; then
				continue
			fi

			if [[ $cpus -ne 2  && $grid != "large" ]]; then
				continue
			fi

			#echo "$nworkers workers(s), $cpus cpu(s), $grid."
			python nni_benchmark.py --cpus=$cpus --nworkers=$nworkers --grid="$grid"

		done
	done
done
