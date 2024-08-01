for i in pM_*
do
cd $i
 cp ../morpy_fifo.sh ./
 cp ../RMSE.sh ./
 for j in sl*
 do
 cp ../SDD_M.CSV $j
 cp ../fit40.py $j
 done
cd .. 
done

