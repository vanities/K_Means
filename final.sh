#!/bin/bash

# iris
#cluster_max=140
# cancer
cluster_max=95

#out="iris-results.txt"
out="cancer-results.txt"

echo "beginning final.."
for ((cluster=1; cluster<=$cluster_max; cluster++)); do
	./script.sh "$cluster"
	#awk '{ total += $1 } END { print total/NR }' temp >> $out
	echo "finished cluster#$cluster"
	#awk  '{ printf( "%s ", $1 ); } END { printf( "\n" ); }' out$1  >> $out
done


#paste out{1..140} > $out 
paste out{1..95} > $out 


rm out*
echo "finished."