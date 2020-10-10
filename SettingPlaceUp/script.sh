cd house
names=`ls`
for eachName in $names; do
mv ./$eachName ../Files/house_$eachName
done
cd ..

cd stop
names=`ls`
for eachName in $names; do
mv ./$eachName ../Files/stop_$eachName
done
cd ..

cd wow
names=`ls`
for eachName in $names; do
mv ./$eachName ../Files/wow_$eachName
done
cd ..