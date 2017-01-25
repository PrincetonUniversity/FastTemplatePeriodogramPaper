name=paper

all : 
	pdflatex $(name).tex && open $(name).pdf

clean :
	rm -f *log *aux *bbl
