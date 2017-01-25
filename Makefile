name=paper

all : 
	pdflatex $(name)

clean :
	rm -f *log *aux *bbl $(name).pdf
