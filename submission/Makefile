name=paper

all : 
	pdflatex $(name)
	bibtex $(name)
	pdflatex $(name)
	pdflatex $(name)

clean :
	rm -f *log *aux *bbl *blg *brf *out $(name)Notes.bib
