import argparse
from image_to_3d import process_image
#from text_to_3d import process_text

def main():
    parser =argparse.ArgumentParser(description="Convert image to 3D point cloud.")
    parser.add_argument("--input_type",choices=["image","text"],required =True,help ="Type of input")
    parser.add_argument("--input",required =True,help ="path to image or text prompt")
    args =parser.parse_args()

    if args.input_type =="image":
        process_image(args.input)
    elif args.input_type =="text":
        #process_text(args.input) 
        print("Text to 3D processing is not implemented yet.")
        

if __name__ == "__main__":
    main()
