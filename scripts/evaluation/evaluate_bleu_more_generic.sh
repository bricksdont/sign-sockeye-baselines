#! /bin/bash

# calling script needs to set

# $hyp
# $ref
# $output
# $tokenize

if [[ $tokenize == "true" ]]; then
    tokenize_arg=""
else
    tokenize_arg="--tokenize none"
fi

for unused in pseudo_loop; do

    if [[ -s $output ]]; then
      continue
    fi

    cat $hyp | sacrebleu $ref -w 3 $tokenize_arg > $output

    echo "$output"
    cat $output

done
