name=proceedings_fast_template_periodogram

all : 
	pdflatex $(name)
	bibtex $(name)
	pdflatex $(name)
	pdflatex $(name)

clean :
	rm -f *log *aux *bbl *blg *brf *out $(name)Notes.bib
