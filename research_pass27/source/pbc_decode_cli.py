import sys
from pbc_cli import main

def decoder_main():
    if len(sys.argv)>1 and sys.argv[1] not in ('inspect','benchdecode','--version','--help','-h'):
        sys.argv.insert(1,'decode')
    return main(decoder_only=True)

if __name__=='__main__': sys.exit(decoder_main())
