import hashlib,json,pathlib,platform,subprocess,sys
from PIL import Image,ImageOps
R=pathlib.Path(__file__).resolve().parents[2]
def git(path,*args):
 p=subprocess.run(['C:/Program Files/Git/cmd/git.exe','-C',str(path),*args],capture_output=True,text=True,check=True)
 return p.stdout.strip()
assets=[]
for name,role in [('googlewiki.png','development: text and sharp graphics'),('mario64.png','development: synthetic graphics'),('flower.jpg','development: smooth photography'),('cat.jpg','development: detailed photography'),('coast.jpg','development: large natural photograph'),('cathedral.png','held-out'),('park.jpg','held-out'),('ducklings.jpg','held-out')]:
 p=R/'hpt_data'/name
 with Image.open(p) as im: w,h=ImageOps.exif_transpose(im).size
 b=p.read_bytes(); assets.append({'name':name,'role':role,'width':w,'height':h,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
repos={
 'libaom':{'origin':'https://aomedia.googlesource.com/aom','tag':'v3.12.1','tag_object':'fc5cf6a132697487fbaa9965b249012e0238768f8','commit':'10aece4157eb79315da205f39e19bf6ab3ee30d0','license':'BSD 2-Clause plus Alliance for Open Media patent license'},
 'libavif':{'origin':'https://github.com/AOMediaCodec/libavif.git','tag':'v1.3.0','tag_object':'69847e00cd7ce09a59112d78dbe4ac167216e49e','commit':'1aadfad932c98c069a1204261b1856f81f3bc199','license':'BSD 2-Clause; LICENSE includes third-party notices'}}
deps=[]
for name,rel in [('libpng','build/libavif-reference/_deps/libpng-src'),('zlib','build/libavif-reference/_deps/zlib-src'),('libyuv','build/libavif-reference/_deps/libyuv-src'),('libwebp/libsharpyuv','build/libavif-reference/_deps/libwebp-src'),('libjpeg-turbo','build/libavif-reference/libjpeg/src/libjpeg')]:
 p=R/'research_pass35'/rel
 try: deps.append({'name':name,'path':rel,'commit':git(p,'rev-parse','HEAD')})
 except Exception as e: deps.append({'name':name,'path':rel,'revision_error':str(e)})
manifest={'repository':{'name':'EgeEken/PBC','branch':'research/experiments','starting_head':'3723dc14de88c6c3dcbd98dd80a097e019a92a9f'},'upstream_sources':repos,'transitive_dependencies':deps,'assets':assets,'environment':{'os':platform.platform(),'python':sys.version.split()[0],'pillow':__import__('PIL').__version__,'numpy':__import__('numpy').__version__,'cpu':platform.processor(),'logical_cpus':__import__('os').cpu_count(),'cmake':'3.29.2','ninja':'1.12.0','gcc':'13.2.0 (Strawberry MinGW-w64 x86_64 UCRT)','nasm':'available','msvc':'unavailable'},'builds':{'reference_libaom':'research_pass35/install/reference/lib/libaom.a','research_libaom_instrumentation_off':'research_pass35/install/research-off/lib/libaom.a','research_libaom_profile':'research_pass35/install/profile/lib/libaom.a','libavif_apps':'research_pass35/build/libavif-reference/avifenc.exe and avifdec.exe','static_link_verified':True,'instrumentation':'CONFIG_COLLECT_PARTITION_STATS=2; CONFIG_COLLECT_COMPONENT_TIMING=1; PBC_PROFILE guarded counters'}}
(R/'research_pass35'/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'assets':assets,'dependencies':deps},indent=2))
