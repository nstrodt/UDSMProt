#mkdir data

# $1: year (2017)
# $2: month (03)

# Download Swissprot $1 $2
curl ftp://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-$1_$2/knowledgebase/uniprot_sprot-only$1_$2.tar.gz -o data/uniprot_sprot-only$1_$2.tar.gz

# Download Uniref $1 $2
curl ftp://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-$1_$2/uniref/uniref$1_$2.tar.gz -o data/uniref$1_$2.tar.gz

# Extract and rename
cd data

# Swissprot
tar -xf uniprot_sprot-only$1_$2.tar.gz uniprot_sprot.xml.gz
gunzip uniprot_sprot.xml.gz
mv uniprot_sprot.xml uniprot_sprot_$1_$2.xml 

# Uniref
tar -xf uniref$1_$2.tar.gz uniref50.xml.gz
gunzip uniref50.xml.gz
mv uniref50.xml uniref50_$1_$2.xml 