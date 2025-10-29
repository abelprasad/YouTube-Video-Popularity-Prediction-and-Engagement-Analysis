import markdown
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib import colors
from bs4 import BeautifulSoup, NavigableString, Tag
import os
import re

def process_table(table_element):
    data = []
    for row in table_element.find_all('tr'):
        row_data = []
        for cell in row.find_all(['th', 'td']):
            cell_text = cell.get_text().strip()
            row_data.append(cell_text)
        if row_data:
            data.append(row_data)

    if not data:
        return None

    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWHEIGHT', (0, 0), (-1, -1), 20),
    ])

    pdf_table = Table(data, hAlign='LEFT')
    pdf_table.setStyle(table_style)
    return pdf_table

def create_pdf_from_markdown(md_file, pdf_file):
    print(f"Reading markdown from: {md_file}")

    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    print("Converting markdown to HTML...")
    html_content = markdown.markdown(
        md_content,
        extensions=['extra', 'tables', 'fenced_code', 'codehilite']
    )

    print("Creating PDF...")
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='Justify',
        parent=styles['BodyText'],
        alignment=TA_JUSTIFY,
        fontSize=10,
        leading=14
    ))
    styles.add(ParagraphStyle(
        name='CodeBlock',
        parent=styles['Code'],
        fontSize=8,
        leading=10,
        leftIndent=20,
        rightIndent=20,
        backColor=colors.lightgrey
    ))

    story = []
    soup = BeautifulSoup(html_content, 'html.parser')

    for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table', 'pre', 'code', 'hr']):
        try:
            if element.name == 'hr':
                story.append(Spacer(1, 0.2*inch))

            elif element.name == 'h1':
                story.append(Spacer(1, 0.3*inch))
                text = element.get_text().strip()
                story.append(Paragraph(f'<b><font size="18">{text}</font></b>', styles['Heading1']))
                story.append(Spacer(1, 0.2*inch))

            elif element.name == 'h2':
                story.append(Spacer(1, 0.2*inch))
                text = element.get_text().strip()
                story.append(Paragraph(f'<b><font size="16">{text}</font></b>', styles['Heading2']))
                story.append(Spacer(1, 0.15*inch))

            elif element.name == 'h3':
                story.append(Spacer(1, 0.15*inch))
                text = element.get_text().strip()
                story.append(Paragraph(f'<b><font size="14">{text}</font></b>', styles['Heading3']))
                story.append(Spacer(1, 0.1*inch))

            elif element.name in ['h4', 'h5', 'h6']:
                story.append(Spacer(1, 0.1*inch))
                text = element.get_text().strip()
                story.append(Paragraph(f'<b><font size="12">{text}</font></b>', styles['Heading4']))
                story.append(Spacer(1, 0.08*inch))

            elif element.name == 'table':
                pdf_table = process_table(element)
                if pdf_table:
                    story.append(Spacer(1, 0.1*inch))
                    story.append(pdf_table)
                    story.append(Spacer(1, 0.15*inch))

            elif element.name == 'pre':
                code_text = element.get_text()
                lines = code_text.split('\n')
                story.append(Spacer(1, 0.1*inch))
                for line in lines:
                    if line.strip():
                        story.append(Preformatted(line, styles['Code']))
                story.append(Spacer(1, 0.1*inch))

            elif element.name == 'code' and element.parent.name != 'pre':
                text = element.get_text().strip()
                parent_text = element.parent.get_text()
                full_text = parent_text.replace(element.get_text(), f'<font name="Courier" size="9">{text}</font>')
                story.append(Paragraph(full_text, styles['BodyText']))
                story.append(Spacer(1, 0.08*inch))

            elif element.name == 'p':
                text = element.get_text().strip()
                if text:
                    html_text = str(element)
                    html_text = html_text.replace('<strong>', '<b>').replace('</strong>', '</b>')
                    html_text = html_text.replace('<em>', '<i>').replace('</em>', '</i>')
                    html_text = re.sub(r'<code>(.*?)</code>', r'<font name="Courier" size="9">\1</font>', html_text)
                    html_text = re.sub(r'<a [^>]*>(.*?)</a>', r'\1', html_text)
                    html_text = re.sub(r'</?p>', '', html_text)

                    try:
                        story.append(Paragraph(html_text, styles['BodyText']))
                    except:
                        story.append(Paragraph(text, styles['BodyText']))
                    story.append(Spacer(1, 0.08*inch))

        except Exception as e:
            print(f"Warning: Error processing element {element.name}: {str(e)}")
            continue

    print("Building PDF...")
    try:
        doc.build(story)
        print(f"[OK] PDF created successfully: {pdf_file}")

        file_size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
        print(f"     File size: {file_size_mb:.2f} MB")

        return True
    except Exception as e:
        print(f"[ERROR] Failed to create PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    md_file = 'reports/final_report.md'
    pdf_file = 'reports/final_report.pdf'

    if not os.path.exists(md_file):
        print(f"[ERROR] Markdown file not found: {md_file}")
        exit(1)

    success = create_pdf_from_markdown(md_file, pdf_file)

    if success:
        print("\nConversion completed successfully!")
    else:
        print("\nConversion failed!")
        exit(1)
