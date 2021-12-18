# shortcut script to update table of contents in README.md

# if first time, uncomment below install line:
#sudo npm install -g doctoc

# update only pre-existing tocs
doctoc --title '# Table of Contents' -u .

# update main README with just top level
doctoc --title '# Table of Contents' --maxlevel 1 README.md

