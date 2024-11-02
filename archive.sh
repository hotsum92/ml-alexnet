mkdir $1

./view.sh

mv *.png $1
mv *.csv $1
cp run.sh $1
