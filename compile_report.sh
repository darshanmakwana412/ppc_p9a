wget https://github.com/Wandmalfarbe/pandoc-latex-template/releases/download/v3.2.0/Eisvogel-3.2.0.tar.gz -O eisvogel.tar.gz
tar -xzvf eisvogel.tar.gz
rm -rf eisvogel.tar.gz

pandoc README.md \
  -o report.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V colorlinks=true \
  -V linkcolor=RoyalBlue \
  -V mainfont="TeX Gyre Pagella" \
  -V fontsize=11pt