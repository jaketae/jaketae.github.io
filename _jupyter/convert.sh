nb=$1

function convert(){
	jupyter nbconvert $nb --to markdown
	sed -i'' -e '1s;^;$---\ntitle: TITLE\nlast_modified_at: 2019-12-01 9:50:00\ncategories:\ntags:\n---\n;' ${nb%.ipynb}.md
	mv ${nb%.ipynb}.md ../_posts/
	mv ${nb%.ipynb}_files ../assets/images/
	echo "Conversion complete!"
}

convert