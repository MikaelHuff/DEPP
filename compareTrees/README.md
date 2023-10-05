# compareTrees

The MicrobesOnline tree comparison tool

run as follows:

`compareTrees.missingBranch tree1.nwk tree2.nwk -simplify`

which reports

| number of bipartitions | number of false negative bipartitions | the proportion of false negative bipartitions |
|------------------------|---------------------------------------|-----------------------------------------------|

When run with `-simplify` the number of false negative bipartitions is equal to half of RF (Robinson and Foulds) distance

if internal nodes have labels, the command above might fail. The simplest solution is:

`compareTrees.missingBranch <(nw_topology -bI tree1.nwk) <(nw_topology -bI tree2.nwk) -simplify`

where `nw_topology` is a command in newick utilites.


