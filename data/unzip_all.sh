for i in zips/*.zip 
do 
    filename=$(basename "$i")
    unzip -q "$i" -d "raws/${filename%%.zip}"
done
