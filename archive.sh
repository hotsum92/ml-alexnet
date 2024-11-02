if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <folder>"
    exit 1
fi

mkdir $1

./view.sh

mv *.png $1
mv *.csv $1
cp run.sh $1
