# get current version
VERSION_=`cat VERSION.txt`

# increment
VERSION=$((VERSION+1))

echo "Changing v${VERSION_} -> v${VERSION}"

# change file
echo $VERSION > VERSION.txt

# change paper
cp paper_v${VERSION_}.tex paper_${VERSION}.tex

# move (now-)old paper to old_versions
mv paper_v${VERSION_}.tex old_versions/

# soft-link paper.tex -> (new paper)
ln -s paper_v${VERSION}.tex paper.tex
