rm MINMSE.dat
for Mor in MorUlist MorElist MorIPlist 
do
rm rmse_${Mor}.dat
  for sl in sl*
  do
  cd $sl
    for dat in sl*.dat
    do 
    rmse=`grep Ave $dat|grep $Mor|awk '{print $6}'`
    r2=`grep Ave $dat|grep $Mor|awk '{print $4}'`
    r=`grep Ave $dat|grep $Mor|awk '{print $8}'`
    nsl=${sl#sl}
    ncol1=${dat%.dat}
    ncol=${ncol1#*_}
    printf "$nsl $ncol $rmse $r2 $r\n" >>../rmse_${Mor}.dat
    done
  cd ..
  done
cat rmse_${Mor}.dat|sort -nk1 -nk2 >sort_rmse_${Mor}.dat
echo ${Mor} >>MINMSE.dat
cat rmse_${Mor}.dat|sort -nk3|head >>MINMSE.dat
rm rmse_${Mor}.dat
done
