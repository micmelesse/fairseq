PWD=`pwd`
DIR_NAME=`basename $PWD`
echo $DIR_NAME
rsync -av . -e "ssh -p 20059" michael@10.216.64.100:~/$DIR_NAME

