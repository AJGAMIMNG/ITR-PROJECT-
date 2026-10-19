import os
import io
import zipfile
from PIL import Image
import xml.etree.ElementTree as ET

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "captured_screenshots")
SRC_DOCX = os.path.join(BASE_DIR, "temp_report_read.docx")
DST_DOCX = os.path.join(os.path.dirname(BASE_DIR), "ITRREPORT_Student_Performance_Final_v2.docx")

# Define the 7 genuine replacement images
images_config = [
    {
        "target_name": "word/media/image1.jpeg",
        "rid": "rId7",
        "src_path": os.path.join(SCREENSHOTS_DIR, "screen1_dashboard.png"),
        "caption": "Figure 1: Gradewatch Platform Executive Overview Dashboard & Academic KPI Cards"
    },
    {
        "target_name": "word/media/image2.jpeg",
        "rid": "rId8",
        "src_path": os.path.join(SCREENSHOTS_DIR, "screen2_inputs.png"),
        "caption": "Figure 2: Student Profile Evaluation & Feature Input Interface"
    },
    {
        "target_name": "word/media/image3.jpeg",
        "rid": "rId9",
        "src_path": os.path.join(SCREENSHOTS_DIR, "screen_terminal_cli.png"),
        "caption": "Figure 3: Terminal CLI Execution of report_code.py (ML Training, Evaluation & Prediction)"
    },
    {
        "target_name": "word/media/image4.jpeg",
        "rid": "rId10",
        "src_path": os.path.join(SCREENSHOTS_DIR, "screen7_diagnostics.png"),
        "caption": "Figure 4: 5-Model Performance Comparison Matrix (Random Forest, Gradient Boosting, SVR, etc.)"
    },
    {
        "target_name": "word/media/image5.jpeg",
        "rid": "rId11",
        "src_path": os.path.join(SCREENSHOTS_DIR, "screen5_whatif.png"),
        "caption": "Figure 5: What-If Scenario & Academic Intervention Simulation Sandbox"
    },
    {
        "target_name": "word/media/image6.jpeg",
        "rid": "rId12",
        "src_path": os.path.join(SCREENSHOTS_DIR, "screen6_roster.png"),
        "caption": "Figure 6: Classroom Roster & Cohort Performance Intelligence Dashboard"
    },
    {
        "target_name": "word/media/image7.jpeg",
        "rid": "rId13",
        "src_path": os.path.join(SCREENSHOTS_DIR, "model_evaluation_plot.png"),
        "caption": "Figure 7: Machine Learning Model Diagnostics (Actual vs. Predicted G3 & Feature Importances)"
    }
]

# Process and convert each image to JPEG bytes and calculate EMU dimensions
prepared_images = {}
emu_dimensions = {}  # rid -> (cx, cy)
PAGE_WIDTH_EMU = 5731510  # 6.27 inches

for cfg in images_config:
    src_p = cfg["src_path"]
    if not os.path.exists(src_p):
        raise FileNotFoundError(f"Missing source screenshot: {src_p}")

    img = Image.open(src_p).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    prepared_images[cfg["target_name"]] = buf.getvalue()

    cx = PAGE_WIDTH_EMU
    cy = int(PAGE_WIDTH_EMU * (img.height / img.width))
    # Cap maximum height to ~4.5 inches (4,114,800 EMU) to fit comfortably on page
    if cy > 4114800:
        cy = 4114800
        cx = int(4114800 * (img.width / img.height))

    emu_dimensions[cfg["rid"]] = (cx, cy)
    print(f"Processed {cfg['target_name']}: {img.width}x{img.height} -> EMU {cx}x{cy}")

# Open source docx
zin = zipfile.ZipFile(SRC_DOCX, "r")
doc_xml = zin.read("word/document.xml")

ET.register_namespace("w", "http://schemas.openxmlformats.org/wordprocessingml/2006/main")
ET.register_namespace("w14", "http://schemas.microsoft.com/office/word/2010/wordml")
ET.register_namespace("w15", "http://schemas.microsoft.com/office/word/2012/wordml")
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace("m", "http://schemas.openxmlformats.org/officeDocument/2006/math")
ET.register_namespace("v", "urn:schemas-microsoft-com:vml")
ET.register_namespace("wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing")
ET.register_namespace("a", "http://schemas.openxmlformats.org/drawingml/2006/main")
ET.register_namespace("pic", "http://schemas.openxmlformats.org/drawingml/2006/picture")

tree = ET.fromstring(doc_xml)
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

# Find drawings and update extents according to their blip rid
for inline in tree.findall(f".//{{{WP_NS}}}inline"):
    blip = inline.find(f".//{{{A_NS}}}blip")
    if blip is not None:
        rid = blip.attrib.get(f"{{{R_NS}}}embed")
        if rid in emu_dimensions:
            cx, cy = emu_dimensions[rid]
            # Update wp:extent
            extent = inline.find(f"{{{WP_NS}}}extent")
            if extent is not None:
                extent.attrib["cx"] = str(cx)
                extent.attrib["cy"] = str(cy)
            # Update a:xfrm/a:ext
            xfrm = inline.find(f".//{{{A_NS}}}xfrm")
            if xfrm is not None:
                ext = xfrm.find(f"{{{A_NS}}}ext")
                if ext is not None:
                    ext.attrib["cx"] = str(cx)
                    ext.attrib["cy"] = str(cy)

# Add captions below each image paragraph
rid_to_caption = {cfg["rid"]: cfg["caption"] for cfg in images_config}
body = tree.find(f".//{{{W_NS}}}body")
children = list(body)

# Track paragraphs to insert captions
for i, p in enumerate(children):
    blip = p.find(f".//{{{A_NS}}}blip")
    if blip is not None:
        rid = blip.attrib.get(f"{{{R_NS}}}embed")
        if rid in rid_to_caption:
            cap_text = rid_to_caption[rid]
            # Center the image paragraph itself
            pPr = p.find(f"{{{W_NS}}}pPr")
            if pPr is None:
                pPr = ET.Element(f"{{{W_NS}}}pPr")
                p.insert(0, pPr)
            jc = pPr.find(f"{{{W_NS}}}jc")
            if jc is None:
                jc = ET.SubElement(pPr, f"{{{W_NS}}}jc")
            jc.attrib[f"{{{W_NS}}}val"] = "center"

            # Check next paragraph
            if i + 1 < len(children):
                next_p = children[i + 1]
                # If next_p is empty, use it for caption
                if not "".join(next_p.itertext()).strip():
                    next_pPr = next_p.find(f"{{{W_NS}}}pPr")
                    if next_pPr is None:
                        next_pPr = ET.Element(f"{{{W_NS}}}pPr")
                        next_p.insert(0, next_pPr)
                    next_jc = next_pPr.find(f"{{{W_NS}}}jc")
                    if next_jc is None:
                        next_jc = ET.SubElement(next_pPr, f"{{{W_NS}}}jc")
                    next_jc.attrib[f"{{{W_NS}}}val"] = "center"

                    r = ET.SubElement(next_p, f"{{{W_NS}}}r")
                    rPr = ET.SubElement(r, f"{{{W_NS}}}rPr")
                    ET.SubElement(rPr, f"{{{W_NS}}}rFonts", {
                        f"{{{W_NS}}}ascii": "Times New Roman",
                        f"{{{W_NS}}}hAnsi": "Times New Roman",
                        f"{{{W_NS}}}cs": "Times New Roman"
                    })
                    ET.SubElement(rPr, f"{{{W_NS}}}i")
                    ET.SubElement(rPr, f"{{{W_NS}}}b")
                    ET.SubElement(rPr, f"{{{W_NS}}}sz", {f"{{{W_NS}}}val": "22"})
                    ET.SubElement(rPr, f"{{{W_NS}}}szCs", {f"{{{W_NS}}}val": "22"})
                    t = ET.SubElement(r, f"{{{W_NS}}}t")
                    t.text = cap_text

new_doc_xml = ET.tostring(tree, encoding="utf-8", xml_declaration=True)

# Write out final v2 docx
zout = zipfile.ZipFile(DST_DOCX, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    if item.filename == "word/document.xml":
        zout.writestr(item, new_doc_xml)
    elif item.filename in prepared_images:
        zout.writestr(item, prepared_images[item.filename])
    else:
        zout.writestr(item, zin.read(item.filename))

zout.close()
zin.close()

print(f"Successfully generated {DST_DOCX} (Size: {os.path.getsize(DST_DOCX)} bytes)")
