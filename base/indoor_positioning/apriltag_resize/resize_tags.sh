#!/bin/bash

# ==============================================
# User inputs (change these settings!)
# ==============================================

# Set the folder to use
FILES=tag36h11/*.png  # Set the folder to convert from

# Set the desired width of your tags
width=120.0         # [mm]  Set the desired width in real units
dpi=500            # [ppi] Set the dpi (to control the print quality)
extra_margin=0     # [px]  Add an extra margin around your tags
default_width=10.0 # [px]  The pixel width of the original images

# Set the grid size you want to make
grid_h=1           # [num] Number of rows
grid_w=2           # [num] Number of columns
grid_filetype=pdf  # [png, jpg, pdf] Set the filetype for the grids

# Decide whether to keep intermediate files
keep_single_tag_images=false



# ==============================================
# Do some magick! (ImageMagick)
# ==============================================

# Get the total number of files
total_files=0

for f in $FILES
do
	total_files=$((total_files+1))
done

total_files_check=$((total_files-1))


# Calculate the scale factors
scale1=$(awk "BEGIN {printf \"%.2f\",${width}*(1+(2/${default_width})) * ${dpi} / 24.5 * 100 / ${default_width}"})
scale=$(awk "BEGIN {printf \"%d\", ${scale1}"})

pixels1=$(awk "BEGIN {printf \"%.2f\",${width}*(1+(2/${default_width})) * ${dpi} / 24.5"})
pixels=$(awk "BEGIN {printf \"%d\", ${pixels1}"})

margin=$(awk "BEGIN {printf \"%d\", ${pixels1}* ${extra_margin}/${default_width}"})

grid_check=$((grid_h * grid_w))

echo "$scale1"
echo "$scale"
echo "$margin"


# Make a fresh output directory
base="$(dirname -- $FILES[0])"
OUTFOLDER="${base}/${width}mm"
mkdir -p $OUTFOLDER
rm -r $OUTFOLDER/*


# Set up some variables
allnames=""
i=0
last_i=0
montage_num=0

# Loop over all pictures
for f in $FILES
do
	# Skip the mosaic picture
	if [[ "$f" == *"mosaic"* ]]; then
		total_files_check=$((total_files_check-1))
		continue
	fi

	# Generate the output filename for the current image
	ext="${f#*.}"
	filename="$(basename -- $f)"
	name=$(echo "$filename" | cut -f 1 -d '.')
	outname="${OUTFOLDER}/${name}.${ext}"
	#echo "$outname"

	# Rescale the current image
	convert $f -scale $scale% $outname

	# Add the image to the list of images to add to the current grid
	allnames="${allnames} ${outname}"

	# Every time we need to generate a grid, do it 
	v=$(((i+1)%grid_check))
    if [[ ( "$v" -eq 0  &&  "$i" != 0 ) || "$i" == "$total_files_check" ]]; then

    	# Generate the output filename for the grid
    	last_i_str=`printf "%05d\n" $last_i`
    	i_str=`printf "%05d\n" $i`
    	outmontage="${OUTFOLDER}/grid_${last_i_str}_${i_str}.${grid_filetype}"

    	# Generate the grid and save it as a pdf
		montage $allnames -geometry ${pixels}x${pixels}+${margin}+${margin} -tile ${grid_w}x -density ${dpi} $outmontage
		echo "$outmontage"

		# Update grid counters
		last_i=$((i+1))
		montage_num=$((montage_num+1))
    	allnames=""
    fi

    # Update single image counter
	i=$((i+1))
done

# Combine all single-page PDFs into one
if [ "$grid_filetype" == "pdf" ]; then
	echo "Combining PDFs"
	sleep 1
	convert -density ${dpi} ${OUTFOLDER}/*.pdf ${OUTFOLDER}/all_tags.pdf
	rm -r $OUTFOLDER/grid*
fi

if [ "$keep_single_tag_images" != true ]; then
	rm -r $OUTFOLDER/*.png
fi