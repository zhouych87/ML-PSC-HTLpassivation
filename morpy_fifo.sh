[ -e ./fd1 ] || mkfifo ./fd1
exec 6<>./fd1      
rm -rf ./fd1   
for((i=1;i<=20;i++))
do
echo >&6
done

for col in $(seq 1 1 41)
do
  for sl in sl*
  do
  read -u6
  {
  cd $sl
  row=`sed -n '/colseq=/=' fit40.py`
  sed -i "${row}c \    colseq=$col" fit40.py
  python fit40.py > ${sl}_${col}.dat
  cd ..
  echo >&6
  }&
  done
done
wait

exec 6<&-                       #关闭文件描述符的读
exec 6>&-                       #关闭文件描述符的写

bash RMSE.sh
for i in sort*
do
hmap.py $i sl nsc
done
