
# $1 true tree
# $2 placement tree
# $3 query sequence
# $4 backbone tree

temp_dir=$(mktemp -d)
#echo ${temp_dir}
nw_labels -I $4 > ${temp_dir}/backbone.txt
grep ">" $3 | sed "s/>//g" > ${temp_dir}/query.txt
while read -r line
do
	echo $line > ${temp_dir}/query
	cat ${temp_dir}/backbone.txt ${temp_dir}/query > ${temp_dir}/cur_label.txt
	mapfile -t < ${temp_dir}/cur_label.txt
	new_query=${line//_[^ \t]*/}
	nw_prune -v $2 "${MAPFILE[@]}" > ${temp_dir}/tmp.nwk
	sed  "s/${line}/${new_query}/g" ${temp_dir}/tmp.nwk > ${temp_dir}/.tmp.nwk
	echo "$line" $( $PWD/compareTrees/compareTrees.missingBranch <(nw_topology $1) <(nw_topology ${temp_dir}/.tmp.nwk) -simplify )
done < ${temp_dir}/query.txt
#rm -rf ${temp_dir}
