name=proceedings_v`cat PROC_VERSION.txt`

all : 
	pdflatex $(name)
	pdflatex $(name)
	bibtex $(name)
	bibtex $(name)
	pdflatex $(name)
	pdflatex $(name)

clean :
	rm -f *log *aux *bbl *blg *brf *out $(name)Notes.bib
