header='name,epoch,time,loss,accuracy'

cat *.csv | grep 'train' | grep -v "$header" | sed '1 i\'"$header" | csvtk plot line -x epoch -y loss -g name > loss.png
cat *.csv | grep 'train' | grep -v "$header" | sed '1 i\'"$header" | csvtk plot line -x epoch -y accuracy -g name > accuracy.png
