# -*- coding: utf-8 -*-
import io
import os
import random  # Used for potential delays
import re
import sys
import time  # Used for potential delays if needed
import bs4
import requests

# --- Global Configuration ---
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


MAX_RETRIES = 2
BASE_RETRY_DELAY = 3
SEC_DOWNLOAD_DIR = '/usr/local/google/home/abhishinde/ml/sec/filings'


# --- Helper Function: Fetch and Parse HTML with Retries ---
# (Takes viewer URL, returns bs4.BeautifulSoup object or None)
def get_and_parse_html_from_viewer(
    viewer_url, retries=MAX_RETRIES, delay=BASE_RETRY_DELAY
):
  """Fetches and parses the underlying iXBRL document from an SEC viewer URL.

  Args:
      viewer_url (str): The URL of the SEC iXBRL viewer (e.g.,
        https://www.sec.gov/ix?doc=...).

  Returns:
      bs4.BeautifulSoup object or None if fetching/parsing fails.
  """
  print(f'Processing URL: {viewer_url}')
  # --- Extract document URL ---
  try:
    doc_path_match = re.search(r'\?doc=(.*)', viewer_url)
    if not doc_path_match:
      raise ValueError("Could not find '?doc=' parameter.")
    doc_path = doc_path_match.group(1)
    if not doc_path:
      raise ValueError("Document path after '?doc=' is empty.")
    document_url = f'https://www.sec.gov{doc_path}'
    print(f'  Extracted direct document URL: {document_url}')
  except Exception as e:
    print(
        'Error: Could not extract document path from viewer URL:'
        f' {viewer_url}\n{e}',
        file=sys.stderr,
    )
    return None

  # --- Fetch document ---
  for attempt in range(retries + 1):
    wait_time = delay + random.uniform(0, 1)
    try:
      print(f'  Fetching direct document (Attempt {attempt + 1})')
      response = requests.get(document_url, headers=HEADERS, timeout=60)
      response.raise_for_status()
      if response.content.startswith(b'\xef\xbb\xbf'):
        html_content = response.content.decode('utf-8-sig')
      else:
        html_content = response.content.decode('utf-8')
      print('  Successfully fetched document.')

      # --- Parse document ---
      print('  Parsing HTML/iXBRL content...')
      try:
        soup = bs4.BeautifulSoup(html_content, 'lxml')  # Primary parser
        if not soup.body:
          raise ValueError('Parsed document missing <body> tag.')
        print('  Successfully parsed document.')
        return soup
      except Exception as e_lxml:
        print(
            f'  Warning: lxml parser failed ({e_lxml}), falling back to'
            ' html.parser...'
        )
        try:
          soup = bs4.BeautifulSoup(
              html_content, 'html.parser'
          )  # Fallback parser
          if not soup.body:
            raise ValueError('Parsed document (fallback) missing <body> tag.')
          print('  Successfully parsed document using fallback parser.')
          return soup
        except Exception as e_fb:
          print(
              f'  Error parsing HTML with html.parser fallback: {e_fb}',
              file=sys.stderr,
          )
          return None  # Critical failure

    except requests.exceptions.Timeout:
      print(f'  Timeout error fetching {document_url}.')
    except requests.exceptions.HTTPError as http_err:
      print(f'  HTTP error fetching {document_url}: {http_err}')
      if 400 <= http_err.response.status_code < 500:
        break
    except requests.exceptions.RequestException as req_err:
      print(f'  Request error fetching {document_url}: {req_err}')
    except Exception as e:
      print(f'  An unexpected error occurred during fetch/decode: {e}')
      break

    if attempt < retries:
      print(f'  Retrying in {wait_time:.2f} seconds...')
      time.sleep(wait_time)
    else:
      print(f'  Failed to fetch {document_url} after {retries + 1} attempts.')
      return None
  return None


def large_mda_section_finder(target_viewer_url):

  # Call the function to fetch and parse the document
  filing_soup = get_and_parse_html_from_viewer(target_viewer_url)

  if filing_soup is None:
    print(
        '\nCritical error: Failed to retrieve or parse the document.',
        file=sys.stderr,
    )
    sys.exit(1)  # Exit with error code

  print(
      "\nSearching for all 'Management Discussion and Analysis' sections to"
      ' find the largest...'
  )

  # --- Find all potential MD&A Headers ---
  mda_header_tags = []
  mda_pattern = re.compile(
      r'(item\s*2\s*\.?)?\s*management.{1,5}s?\s*discussion\s*(and|&)\s*analysis\b',
      re.IGNORECASE,
  )
  # Search within common tags that might contain headings
  potential_elements = (
      filing_soup.body.find_all(
          ['p', 'div', 'b', 'strong', 'h1', 'h2', 'h3', 'h4']
      )
      if filing_soup.body
      else []
  )
  last_header_text = None
  for tag in potential_elements:
    text = tag.get_text(' ', strip=True)
    if mda_pattern.search(text):
      if text != last_header_text:  # Avoid immediate duplicates
        mda_header_tags.append(tag)
        last_header_text = text
        # print(f"  Found potential MD&A header: '{text[:100]}...'") # Keep output cleaner

  if not mda_header_tags:
    print(
        "\nNo sections matching 'Management Discussion and Analysis' were"
        ' found.'
    )
    sys.exit(0)

  print(
      f'\nFound {len(mda_header_tags)} potential MD&A header(s). Extracting'
      ' content to find the largest...'
  )

  # --- Define Item 3 Pattern ---
  item_3_pattern = re.compile(r'^\s*item\s+3\s*[:.]?', re.IGNORECASE)

  # --- List to store extracted sections ---
  mda_sections_found = []  # Stores {'header': ..., 'content': ...} dictionaries

  # --- Loop through each found MD&A header and extract content ---
  for i, mda_start_tag in enumerate(mda_header_tags):
    current_header_text = mda_start_tag.get_text(' ', strip=True)
    print(f"--- Analyzing MD&A Occurrence #{i+1}: '{current_header_text}' ---")

    content_parts = []
    current_element = mda_start_tag  # Start from the header tag itself

    while True:
      current_element = current_element.find_next()  # Get the very next element

      if current_element is None:
        break  # End of document

      # --- Stopping Conditions ---
      # 1. Stop if we hit the *next* identified MD&A header tag
      next_mda_header_tag = (
          mda_header_tags[i + 1] if i + 1 < len(mda_header_tags) else None
      )
      if next_mda_header_tag and current_element == next_mda_header_tag:
        print('      Stopping extraction before next MD&A header.')
        break

      # 2. Stop if we hit a tag starting with "Item 3"
      if hasattr(current_element, 'name'):  # Check if it's a tag
        element_text = current_element.get_text(
            strip=True
        )  # Get stripped text efficiently
        if item_3_pattern.match(element_text):
          print(
              '      Stopping extraction before Item 3:'
              f" '{element_text[:100]}...'"
          )
          break  # Stop before Item 3

      # --- Text Extraction (with table exclusion) ---
      is_table_or_inside = False
      if hasattr(current_element, 'name'):
        if current_element.name == 'table':
          is_table_or_inside = True
        elif current_element.find_parent('table'):
          is_table_or_inside = True
      elif not hasattr(current_element, 'name') and current_element.find_parent(
          'table'
      ):
        is_table_or_inside = True

      if not is_table_or_inside:
        if hasattr(current_element, 'name') and current_element.name not in [
            'script',
            'style',
            'head',
            'meta',
            'title',
        ]:
          style = current_element.get('style', '').replace(' ', '')
          is_hidden = 'display:none' in style or 'visibility:hidden' in style
          if not is_hidden:
            text = current_element.get_text(separator=' ', strip=True)
            if text:
              content_parts.append(text)
        elif (
            not hasattr(current_element, 'name')
            and current_element.string
            and current_element.string.strip()
        ):
          content_parts.append(current_element.string.strip())

    # --- Store Extracted Content for this Occurrence ---
    if content_parts:
      full_section_text = '\n'.join(content_parts)
      cleaned_section_text = re.sub(r'\n\s*\n', '\n', full_section_text).strip()
      # Add the found section to the list
      mda_sections_found.append(
          {'header': current_header_text, 'content': cleaned_section_text}
      )
      print(
          '    Extracted content length:'
          f' {len(cleaned_section_text)} characters.'
      )
    else:
      print('      No non-table content extracted for this occurrence.')

  # --- Process the results after checking all occurrences ---
  if not mda_sections_found:
    print('\nNo MD&A sections with non-table content were found.')
  else:
    # Find the section with the longest content using max() and a lambda function
    largest_section = max(mda_sections_found, key=lambda s: len(s['content']))

    print(
        f'\nFound {len(mda_sections_found)} potential MD&A section(s). Printing'
        ' the largest.'
    )
    print(
        f"\n--- Largest MD&A Section Found: '{largest_section['header']}' ---"
    )
    print(f"(Content Length: {len(largest_section['content'])} characters)")
    return largest_section['content']

  print(f"\nFinished processing at {time.strftime('%Y-%m-%d %H:%M:%S')}.")
  return 'None'


def save_to_file(
    largest_mda_section,
    cik,
    date_filed,
    form_type,
    year,
    quarter,
    filing_document_full_url,
):
  print(
      f'Saving to file for cik: {cik} date_filed: {date_filed} form_type:'
      f' {form_type} year: {year} quarter: {quarter}'
  )
  full_path = ''
  try:
    # Get the user's home directory path
    home_dir = os.path.expanduser(SEC_DOWNLOAD_DIR)

    year_dir = os.path.join(home_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    quarter_dir = os.path.join(year_dir, quarter)
    os.makedirs(quarter_dir, exist_ok=True)
    file_name = f'{year}_{quarter}_{cik}_{date_filed}_{form_type}.txt'
    full_path = os.path.join(quarter_dir, file_name)

    # Open the file in write mode ('w') and write the content
    # Using 'with' ensures the file is properly closed even if errors occur
    with open(full_path, 'w', encoding='utf-8') as f:
      f.write('URL:' + filing_document_full_url + '\n')
      f.write(largest_mda_section)

    print(f'Successfully wrote content to: {full_path}')
    # Return the full path for confirmation
    return full_path

  except IOError as e:
    print(f'Error writing to file {full_path}: {e}')
    raise  # Re-raise the exception if needed for further handling
  except Exception as e:
    print(f'An unexpected error occurred when writing to file {full_path}: {e}')
    raise  # Re-raise for unexpected issues


def clean_text(text: str) -> str:
  text = remove_table_content_lines(text)
  return remove_consecutive_duplicate_lines(text)


def remove_table_content_lines(text: str) -> str:
  string_io = io.StringIO(text)
  result_lines = []

  for line in string_io:
    if 'Table of Contents' in line and len(line.strip()) <= 20:
      continue
    result_lines.append(line)
  return ''.join(result_lines)


def remove_consecutive_duplicate_lines(text: str) -> str:
  """Removes consecutive duplicate lines from a string.

  Empty lines (lines containing only whitespace) are not considered
  for duplicate removal, meaning consecutive empty lines are preserved,
  and an empty line does not break the sequence of non-empty lines
  being checked for duplication.

  Args:
    text: The input string, potentially multi-line.

  Returns:
    A new string with consecutive duplicate non-empty lines removed.
  """
  if not isinstance(text, str):
    raise TypeError('Input must be a string.')

  # Use io.StringIO to handle universal newlines correctly when iterating
  # This is generally more robust than splitlines() for varied line endings
  string_io = io.StringIO(text)
  result_lines = []
  last_non_empty_line = None

  for line in string_io:
    # Check if the line is effectively empty (contains only whitespace)
    is_empty = not line.strip()

    if is_empty:
      # Always append empty lines, don't update last_non_empty_line
      result_lines.append(line)
    else:
      # Compare with the last *non-empty* line added
      if line != last_non_empty_line:
        result_lines.append(line)
        last_non_empty_line = line  # Update the last non-empty line
      # else: it's a consecutive duplicate non-empty line, so skip it

  # Join the lines back. Since StringIO preserves line endings,
  # simple concatenation is appropriate.
  return ''.join(result_lines)
