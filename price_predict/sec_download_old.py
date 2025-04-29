# -*- coding: utf-8 -*-
import re
import sys
import bs4
import requests

# --- Configuration ---
# The IX viewer URL provided by the user
viewer_url = 'https://www.sec.gov/ix?doc=/Archives/edgar/data/1599407/000121390024100149/ea0220609-10q_1847hold.htm'

# Extract the actual document path from the viewer URL
try:
  # Find the part after '?doc='
  doc_path_match = re.search(r'\?doc=(.*)', viewer_url)
  if not doc_path_match:
    raise ValueError("Could not find '?doc=' parameter in the URL.")
  doc_path = doc_path_match.group(1)
  if not doc_path:
    raise ValueError("Document path after '?doc=' is empty.")
except Exception as e:
  print(
      'Error: Could not extract document path from viewer URL:'
      f' {viewer_url}\n{e}',
      file=sys.stderr,
  )
  sys.exit(1)

# Construct the direct URL to the iXBRL document
document_url = f'https://www.sec.gov{doc_path}'

HEADERS = {
    'User-Agent': (
        'YourAppName/1.0 (your.email@example.com)'
    ),  # *** MUST CUSTOMIZE ***
    'Accept': (
        'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'
    ),
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
}


# --- Helper Function to Identify Potential Section Headers ---
def is_potential_header(tag):
  """Applies heuristics to guess if a tag is a section header."""
  if (
      not tag
      or not hasattr(tag, 'name')
      or tag.name in ['script', 'style', 'head', 'meta']
  ):
    return False, None  # Not a valid tag or non-content tag

  text = tag.get_text(' ', strip=True)
  # Basic filtering for relevance
  if (
      not text or len(text) > 250 or len(text) < 3
  ):  # Filter empty, very long, very short
    return False, None
  # Avoid text that is clearly just a number (often page numbers or table values)
  if text.replace('.', '', 1).isdigit() or text.replace(',', '').isdigit():
    return False, None
  # Avoid text within tables unless it's explicitly a th? Too complex maybe. Let's allow table content for now.

  # Heuristic 1: Standard HTML Headings
  if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
    # Reduce chance of capturing TOC links styled as headings
    link_children = tag.find_all('a', recursive=False)
    if len(link_children) == 1 and link_children[0].get('href', '').startswith(
        '#'
    ):
      # If it's just a single link pointing to an anchor, likely TOC, skip
      # print(f"DEBUG: Skipping potential header (looks like TOC link): {text}")
      return False, None
    return True, text

  # Heuristic 2: Bold/Strong tags or tags containing primarily bold/strong
  is_bold_tag = tag.name in ['b', 'strong']
  # Check if the tag's direct text content looks like a heading
  # Check parent isn't also bold (avoid double counting)
  parent_is_bold = tag.parent and tag.parent.name in ['b', 'strong']

  if is_bold_tag and not parent_is_bold:
    # Avoid if it's likely part of a paragraph (has preceding/following text siblings)
    if isinstance(tag.previous_sibling, str) and tag.previous_sibling.strip():
      return False, None
    if isinstance(tag.next_sibling, str) and tag.next_sibling.strip():
      return False, None
    # Avoid table headers (th) being misinterpreted if they are bold
    if tag.find_parent('th'):
      return False, None
    return True, text

  # Heuristic 3: Styling indicating bold/underline (less reliable, check tag name too)
  style = tag.get('style', '').lower().replace(' ', '')
  # Check for common bold styles, ensure it's likely a block or significant inline element
  is_styled_bold = (
      'font-weight:bold' in style
      or 'font-weight:700' in style
      or 'font-weight:600' in style
  )
  is_styled_underline = 'text-decoration:underline' in style

  # Heuristic 4: Text patterns (All Caps, Item N, PART N) - Often good indicators
  # Require minimum length for all caps to avoid acronyms like 'US'
  is_all_caps = (
      text.isupper() and len(text) > 5 and ' ' in text
  )  # Mostly uppercase, >5 chars, has space
  is_item_part_pattern = (
      re.match(
          r'^\s*(ITEM|PART)\s+[IVX\d]+\s*[:.-]?\s*[\w\s]+', text, re.IGNORECASE
      )
      is not None
  )

  # Combine heuristics - Prioritize Item/Part pattern
  if is_item_part_pattern:
    return True, text

  # Combine others - requires tag to be block-level or potentially significant inline
  if (
      tag.name in ['p', 'div', 'th'] or is_bold_tag
  ):  # Check common block tags or bold tags
    if is_all_caps:
      return True, text
    if is_styled_bold:
      return True, text
    if is_styled_underline:
      return True, text  # Underline less common but possible

  return False, None


# --- Main Logic ---
def main():
  print(f'Fetching iXBRL document from direct URL: {document_url}')
  try:
    response = requests.get(
        document_url, headers=HEADERS, timeout=60
    )  # Longer timeout
    response.raise_for_status()
    # Use UTF-8 which is standard for modern web/XBRL, check if BOM exists
    if response.content.startswith(b'\xef\xbb\xbf'):
      html_content = response.content.decode(
          'utf-8-sig'
      )  # Decode with BOM signature
    else:
      html_content = response.content.decode(
          'utf-8'
      )  # Decode as standard UTF-8

    print('Successfully fetched document.')

  except requests.exceptions.RequestException as e:
    print(f'Error fetching document: {e}', file=sys.stderr)
    sys.exit(1)
  except Exception as e:
    print(f'Error during fetch or decoding: {e}', file=sys.stderr)
    sys.exit(1)

  print('Parsing HTML/iXBRL content using lxml...')
  try:
    soup = bs4.BeautifulSoup(html_content, 'lxml')
  except Exception as e:
    print(f'Error parsing HTML with lxml: {e}', file=sys.stderr)
    # Fallback attempt with html.parser
    print('Falling back to html.parser...')
    try:
      soup = bs4.BeautifulSoup(html_content, 'html.parser')
    except Exception as e_fb:
      print(
          f'Error parsing HTML with html.parser fallback: {e_fb}',
          file=sys.stderr,
      )
      sys.exit(1)

  if not soup.body:
    print(
        'Error: Could not find the <body> tag in the document.', file=sys.stderr
    )
    # Optional: print first few KB of content to see what was received
    # print("\n--- Received Content Start ---")
    # print(html_content[:2048])
    # print("--- End of Content Start ---")
    sys.exit(1)

  print('Identifying sections based on potential headers...')
  sections = []
  # Use find_all to get all tags, then check each one
  # We iterate through the document structure to maintain order
  all_tags = soup.body.find_all(True)  # Find all tags within body

  # Store the tag object associated with each header found to mark boundaries
  header_tags = []
  for tag in all_tags:
    is_header, header_text = is_potential_header(tag)
    if is_header:
      # Avoid adding effectively duplicate headers immediately following each other
      if not header_tags or header_tags[-1]['text'] != header_text:
        header_tags.append({'tag': tag, 'text': header_text})
        print(f"  Found potential section: '{header_text}'")

  if not header_tags:
    print('\nCould not identify any sections based on the defined heuristics.')
    print(
        "Consider inspecting the document's HTML structure and refining the"
        " 'is_potential_header' function."
    )
    sys.exit(0)

  print(f'\n--- Extracted Sections ({len(header_tags)}) ---')

  # Iterate through the identified header tags and extract content between them
  for i, header_info in enumerate(header_tags):
    current_header_tag = header_info['tag']
    current_header_text = header_info['text']

    print(f'\n\n=== SECTION: {current_header_text} ===')

    content_parts = []
    # Start traversal from the element *after* the current header tag
    current_element = current_header_tag

    while True:
      current_element = (
          current_element.find_next()
      )  # Get the very next element in the parse tree

      if current_element is None:  # Reached end of document
        break

      # Determine the start tag of the *next* section, if one exists
      next_section_start_tag = None
      if i + 1 < len(header_tags):
        next_section_start_tag = header_tags[i + 1]['tag']

      # Stop if we've reached the start tag of the next section
      if next_section_start_tag and current_element == next_section_start_tag:
        break

      # Extract text from relevant tags, ignoring script/style etc.
      # Check if it's a tag (and not NavigableString without a name)
      if hasattr(current_element, 'name') and current_element.name not in [
          'script',
          'style',
          'head',
          'meta',
          'title',
      ]:
        # Check for hidden elements (basic check)
        style = current_element.get('style', '').replace(' ', '')
        is_hidden = 'display:none' in style or 'visibility:hidden' in style

        if not is_hidden:
          # Get text, using space as separator, stripping ends
          text = current_element.get_text(separator=' ', strip=True)
          if text:  # Add only if there's actual text content
            content_parts.append(text)
      # Handle NavigableStrings (text nodes directly under parent) that are not just whitespace
      elif (
          not hasattr(current_element, 'name')
          and current_element.string
          and current_element.string.strip()
      ):
        content_parts.append(current_element.string.strip())

    # Combine and clean the text for the section
    # Join parts with a newline; assumes parts roughly correspond to paragraphs/blocks
    full_section_text = '\n'.join(content_parts)
    # Remove excessive blank lines that might result from joining empty parts
    cleaned_section_text = re.sub(r'\n\s*\n', '\n', full_section_text).strip()
    print(cleaned_section_text)

  print('\n--- End of Sections ---')


if __name__ == '__main__':
  main()
