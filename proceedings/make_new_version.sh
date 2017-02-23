# get current version
VERSION_=`cat PROC_VERSION.txt`

# increment
VERSION=$((VERSION+1))

echo "Changing v${VERSION_} -> v${VERSION}"

# change file
echo $VERSION > PROC_VERSION.txt

# change proceedings
cp proceedings_v${VERSION_}.tex proceedings_v${VERSION}.tex

# move (now-)old proceedings to old_versions
mv proceedings_v${VERSION_}.tex old_versions/
mv proceedings_v${VERSION_}.pdf old_versions/

# soft-link proceedings.tex -> (new proceedings)
ln -s proceedings_v${VERSION}.tex proceedings.tex
