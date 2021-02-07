name=paper

all : 
	pdflatex $(name)
	bibtex $(name)
	pdflatex $(name)
	pdflatex $(name)

clean :
	rm -f *log *aux *blg *brf *out $(name)Notes.bib
