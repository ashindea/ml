# g4d CS-util-BUILD-2024-11-30_121351
# blaze run ads/publisher/quality/micro_models/tensorflow/util:sec_download

# -*- coding: utf-8 -*-
import io
import random
import re
import sys
import time
import bs4
import requests
from google3.ads.publisher.quality.micro_models.tensorflow.util.sec_common import clean_text
from google3.ads.publisher.quality.micro_models.tensorflow.util.sec_common import large_mda_section_finder
from google3.ads.publisher.quality.micro_models.tensorflow.util.sec_common import save_to_file

# --- Configuration ---
TARGET_YEAR = '2024'
# Base URL for SEC website (used for constructing filing URLs)
BASE_SEC_URL = 'https://www.sec.gov'
# URL for the EDGAR archives root (used for constructing index page URLs)
ARCHIVES_URL = f'{BASE_SEC_URL}/Archives/'
# URL for the parent directory of the EDGAR full index files
INDEX_BASE_URL = f'{ARCHIVES_URL}edgar/full-index/'
TARGET_FORM_TYPE = '10-Q'
TARGET_FORM_TYPE_10K = '10-K'
# Define the starting character position (0-based index) where the 'Form Type' column begins in crawler.idx.
FORM_TYPE_START_INDEX = 62
MAX_RETRIES = 2
BASE_RETRY_DELAY = 3  # seconds
# Limit the number of filings to process to avoid excessive requests (set to None to process all)
# Processing many filings can take a very long time and puts load on SEC servers.
MAX_FILINGS_TO_PROCESS = 3  # Process only the first 3 found 10-Q filings

# --- User-Agent ---
# IMPORTANT: Customize this with your application name and contact email.
# Using a generic agent is discouraged by the SEC.
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

# --- MD&A Regex Pattern ---
# Pattern for finding the MD&A section header (case-insensitive)
MDA_PATTERN = re.compile(
    r'(item\s*2\s*\.?)?\s*management.{1,5}s?\s*discussion\s*(and|&)\s*analysis\b',  # \b=word boundary
    re.IGNORECASE,
)


# --- Helper Function: Fetch and Parse HTML with Retries ---
def get_and_parse_html(url, retries=MAX_RETRIES, delay=BASE_RETRY_DELAY):
  """Fetches URL content with retries and returns a BeautifulSoup object."""
  for attempt in range(retries + 1):
    wait_time = delay + random.uniform(0, 2)  # Add slight randomness
    try:
      # Use a different indentation for internal prints
      print(f'      Fetching: {url} (Attempt {attempt + 1})')
      response = requests.get(url, headers=HEADERS, timeout=45)
      response.raise_for_status()

      content_type = response.headers.get('content-type', '').lower()
      if (
          'text/html' not in content_type
          and 'application/xhtml+xml' not in content_type
      ):
        print(
            f"      Warning: Content-Type '{content_type}' might not be HTML."
        )

      # Decode using UTF-8 (handle BOM if present)
      if response.content.startswith(b'\xef\xbb\xbf'):
        html_content = response.content.decode('utf-8-sig')
      else:
        html_content = response.content.decode('utf-8')

      if '<html' not in html_content[:1500].lower():
        print(
            f"      Warning: Content does not appear to contain '<html>' tag"
            f' near start.'
        )

      # Try parsing with lxml first
      try:
        soup = bs4.BeautifulSoup(html_content, 'lxml')
      except Exception as e_lxml:
        print(
            f'      Warning: lxml parser failed ({e_lxml}), falling back to'
            ' html.parser...'
        )
        soup = bs4.BeautifulSoup(html_content, 'html.parser')  # Fallback parser

      print(f'      Successfully fetched and parsed: {url}')
      return soup

    except requests.exceptions.Timeout:
      print(f'      Timeout error fetching {url}.')
    except requests.exceptions.HTTPError as http_err:
      print(f'      HTTP error fetching {url}: {http_err}')
      if 400 <= http_err.response.status_code < 500:
        break  # Don't retry client errors
    except requests.exceptions.RequestException as req_err:
      print(f'      Request error fetching {url}: {req_err}')
    except Exception as e:
      print(
          f'      An unexpected error occurred during fetch/parse of {url}: {e}'
      )
      break

    if attempt < retries:
      print(f'      Retrying in {wait_time:.2f} seconds...')
      time.sleep(wait_time)
    else:
      print(f'      Failed to fetch/parse {url} after {retries + 1} attempts.')
      return None
  return None


# --- Helper Function: Find Filing Document URL from Index Page ---
def find_filing_document_url(index_soup):
  """Parses the index page soup to find the link to the main .htm filing document."""
  if not index_soup:
    return None

  doc_table = index_soup.find('table', summary='Document Format Files')
  if not doc_table:
    doc_table = index_soup.find('table')  # Fallback
  if not doc_table:
    print('      Error: Could not find document table on the index page.')
    return None

  rows = doc_table.find_all('tr')
  for row in rows[1:]:  # Skip header row
    cells = row.find_all('td')
    if len(cells) >= 4:
      doc_cell, type_cell = cells[2], cells[3]
      link_tag, doc_type = doc_cell.find('a'), type_cell.get_text(strip=True)
      if link_tag and link_tag.has_attr('href'):
        doc_href = link_tag['href']
        if (doc_type.startswith('10-Q')) and doc_href.lower().endswith(
            ('.htm', '.html')
        ):
          # Construct full URL relative to the base SEC URL
          filing_doc_full_url = BASE_SEC_URL + doc_href
          print(f'      Found filing document link: {filing_doc_full_url}')
          return filing_doc_full_url
  print('      Error: Could not find 10-Q .htm/.html document link in table.')
  return None


# --- Helper Function to Identify Potential Section Headers (iXBRL/HTML) ---
def is_potential_header(tag):
  """Applies heuristics to guess if a tag is a section header."""
  # (Same logic as in the previous iXBRL extraction step)
  if (
      not tag
      or not hasattr(tag, 'name')
      or tag.name in ['script', 'style', 'head', 'meta']
  ):
    return False, None
  text = tag.get_text(' ', strip=True)
  if not text or len(text) > 250 or len(text) < 3:
    return False, None
  if text.replace('.', '', 1).isdigit() or text.replace(',', '').isdigit():
    return False, None
  if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
    link_children = tag.find_all('a', recursive=False)
    if len(link_children) == 1 and link_children[0].get('href', '').startswith(
        '#'
    ):
      return False, None
    return True, text
  is_bold_tag = tag.name in ['b', 'strong']
  parent_is_bold = tag.parent and tag.parent.name in ['b', 'strong']
  if is_bold_tag and not parent_is_bold:
    if isinstance(tag.previous_sibling, str) and tag.previous_sibling.strip():
      return False, None
    if isinstance(tag.next_sibling, str) and tag.next_sibling.strip():
      return False, None
    if tag.find_parent('th'):
      return False, None
    return True, text
  style = tag.get('style', '').lower().replace(' ', '')
  is_styled_bold = (
      'font-weight:bold' in style
      or 'font-weight:700' in style
      or 'font-weight:600' in style
  )
  is_styled_underline = 'text-decoration:underline' in style
  is_all_caps = text.isupper() and len(text) > 5 and ' ' in text
  is_item_part_pattern = (
      re.match(
          r'^\s*(ITEM|PART)\s+[IVX\d]+\s*[:.-]?\s*[\w\s]+', text, re.IGNORECASE
      )
      is not None
  )
  if is_item_part_pattern:
    return True, text
  if tag.name in ['p', 'div', 'th'] or is_bold_tag:
    if is_all_caps:
      return True, text
    if not is_bold_tag and (is_styled_bold or is_styled_underline):
      if tag.find_parent('td') and len(text) < 20:
        return False, None
      return True, text
  return False, None


# --- Helper Function to Extract MD&A from Parsed Filing ---
def extract_ixbrl_mda(filing_soup):
  """Finds the MD&A section within a parsed iXBRL document (BeautifulSoup object)

  and returns its content. Uses the global is_potential_header function.

  Args:
      filing_soup (BeautifulSoup): The parsed soup object of the filing
        document.

  Returns:
      str: The text content of the MD&A section, or an error message string.
  """
  if not filing_soup or not filing_soup.body:
    return 'Error: Invalid or empty BeautifulSoup object provided.'

  # --- Identify Headers ---
  print('      Identifying section headers in filing document...')
  header_tags = []
  all_tags = filing_soup.body.find_all(True)
  last_header_text = None
  for tag in all_tags:
    is_header, header_text = is_potential_header(tag)  # Call global helper
    if is_header:
      if header_text != last_header_text:
        header_tags.append({'tag': tag, 'text': header_text})
        last_header_text = header_text

  if not header_tags:
    return (
        'Error: Could not identify any section headers in the filing document.'
    )
  print(f'      Found {len(header_tags)} potential section headers.')

  # --- Find MD&A Header and Extract Content ---
  mda_content = None
  mda_header = None
  mda_section_index = -1

  for i, header_info in enumerate(header_tags):
    if MDA_PATTERN.search(header_info['text']):  # Use the global MDA_PATTERN
      mda_header = header_info['text']
      mda_section_index = i
      print(f"      Found MD&A header: '{mda_header}' at index {i}")
      break

  if mda_section_index == -1:
    # Provide more debug info if MD&A is not found
    print('      Available headers:')
    for i, h in enumerate(header_tags):
      print(f"        {i}: {h['text']}")
    return 'Error: MD&A section header not found using pattern.'

  # --- Extract content between this header and the next one ---
  print('      Extracting content following MD&A header...')
  content_parts = []
  current_element = header_tags[mda_section_index][
      'tag'
  ]  # Start from the MD&A header tag

  while True:
    current_element = current_element.find_next()
    if current_element is None:
      break  # End of document

    # Determine the start tag of the *next* identified header section
    next_section_start_tag = (
        header_tags[mda_section_index + 1]['tag']
        if mda_section_index + 1 < len(header_tags)
        else None
    )

    # Stop if we hit the next section's header tag
    if next_section_start_tag and current_element == next_section_start_tag:
      break

    # Extract text from relevant tags, ignoring script/style/hidden etc.
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
    # Also capture relevant text nodes (NavigableString)
    elif (
        not hasattr(current_element, 'name')
        and current_element.string
        and current_element.string.strip()
    ):
      content_parts.append(current_element.string.strip())

  if not content_parts:
    return (
        f"Error: Found MD&A header '{mda_header}' but could not extract"
        ' subsequent content.'
    )

  # Combine and clean the extracted text
  full_section_text = '\n'.join(content_parts)
  cleaned_section_text = re.sub(r'\n\s*\n', '\n', full_section_text).strip()
  print('      Successfully extracted MD&A content.')
  return cleaned_section_text  # Return just the content string


# --- Main Execution Logic ---
def main():
  """Main function to fetch SEC index, find 10-Q filings,

  and extract MD&A section from each.
  """
  extract(2024, 'QTR4')


def extract(year, quarter):
  print(f"Script started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
  #   latest_quarter = 'QTR4'  # For 2024 (based on current date being 2025)
  index_file_url = f'{INDEX_BASE_URL}{year}/{quarter}/crawler.idx'
  print(f'STEP 1: Fetching main index file from: {index_file_url}')

  filing_info_list = []
  data_processing_started = False
  line_number = 0

  # --- Fetch and Parse the crawler.idx file ---
  try:
    response = requests.get(index_file_url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    index_content = io.StringIO(response.text)
    print('Successfully fetched index file. Processing lines...')

    for line in index_content:
      line_number += 1
      if not data_processing_started:
        if line.strip().startswith('-----------'):
          data_processing_started = True
        continue  # Skip headers and separator

      if len(line) > FORM_TYPE_START_INDEX:
        try:
          data_substring = line[FORM_TYPE_START_INDEX:].strip()
          parts = data_substring.split()
          if len(parts) >= 4:
            form_type, cik, date_filed, url_path = (
                parts[0],
                parts[1],
                parts[2],
                parts[3],
            )
            if form_type.startswith(
                TARGET_FORM_TYPE_10K
            ) or form_type.startswith(TARGET_FORM_TYPE):
              # Store info including the relative path as requested
              filing_info_list.append({
                  'form_type': form_type,
                  'date_filed': date_filed,
                  'cik': cik,
                  'relative_url_path': (
                      url_path  # Store relative path from index
                  ),
              })
        except Exception as e:
          print(f'Warning: Error processing index line {line_number}: {e}.')
          continue

  except requests.exceptions.RequestException as e:
    print(
        f"FATAL ERROR fetching index file '{index_file_url}': {e}",
        file=sys.stderr,
    )
    sys.exit(1)
  except Exception as e:
    print(f'FATAL ERROR during initial index processing: {e}', file=sys.stderr)
    sys.exit(1)

  if not filing_info_list:
    print(f"\nNo '{TARGET_FORM_TYPE}' filings found in the index file.")
    sys.exit(0)

  print(
      f"\nSTEP 2: Found {len(filing_info_list)} '{TARGET_FORM_TYPE}' filings in"
      ' index.'
  )
  process_limit = (
      MAX_FILINGS_TO_PROCESS
      if MAX_FILINGS_TO_PROCESS is not None
      else len(filing_info_list)
  )
  print(f'Processing the first {process_limit} filings...')

  # --- Process each found 10-Q filing ---
  filings_processed_count = 0
  for filing_info in filing_info_list:
    if filings_processed_count >= process_limit:
      print(f'\nReached processing limit ({process_limit}). Stopping.')
      break

    filings_processed_count += 1
    form_type = filing_info['form_type']
    cik = filing_info['cik']
    date_filed = filing_info['date_filed']
    relative_url_path = filing_info[
        'relative_url_path'
    ]  # The path from crawler.idx

    # Construct URL to the filing's index page (landing page for the filing)
    # index_page_url = f"{ARCHIVES_URL}{relative_url_path}"
    index_page_url = f'{relative_url_path}'  ## fix

    print(
        f'\n--- Processing Filing {filings_processed_count}/{process_limit} ---'
    )
    print(f'  CIK: {cik}, Form type: {form_type}, Date Filed: {date_filed}')
    print(
        f'  Index File Path (relative): {relative_url_path}'
    )  # Print relative path as requested
    print(f'  Index Page URL: {index_page_url}')

    # Add delay before fetching index page
    time.sleep(
        random.uniform(0.5, 1.0)
    )  # Slightly longer delay between filings

    # --- Step 3: Get the actual filing document URL ---
    index_page_soup = get_and_parse_html(index_page_url)
    if not index_page_soup:
      print(
          f'  ERROR: Failed to get/parse index page {index_page_url}. Skipping'
          ' this filing.'
      )
      continue

    filing_document_full_url = find_filing_document_url(index_page_soup)
    if not filing_document_full_url:
      print(
          '  ERROR: Failed to find .htm document link on index page'
          f' {index_page_url}. Skipping.'
      )
      continue

    # Add another delay before fetching the main filing document
    time.sleep(random.uniform(0.5, 1.5))

    print(f'  Fetching the main filing document: {filing_document_full_url}')

    largest_mda_section_raw = large_mda_section_finder(filing_document_full_url)
    largest_mda_section = clean_text(largest_mda_section_raw)
    print('-' * 70)
    # Print the content of the largest section to standard output
    print(f'largest_mda_section:  {len(largest_mda_section)}')
    print('-' * 70)
    print(
        '--- End of Largest MD&A Section downloaded'
        f' {filing_document_full_url}---'
    )
    save_to_file(
        largest_mda_section,
        cik,
        date_filed,
        form_type,
        year,
        quarter,
        filing_document_full_url,
    )

    # # --- Step 4: Fetch, parse, and extract MD&A ---
    # # Fetch and parse the filing document first
    # filing_document_soup = get_and_parse_html(filing_document_full_url)
    # if not filing_document_soup:
    #      print(f"  ERROR: Failed to get/parse filing document {filing_document_full_url}. Skipping MD&A extraction.")
    #      continue

    # # Extract MD&A using the dedicated helper function
    # print("    Extracting Management's Discussion and Analysis...")
    # mda_section_text = extract_ixbrl_mda(filing_document_soup) # Pass soup object

    # # --- Step 5: Print MD&A Output ---
    # print("\n    --- Management Discussion and Analysis ---")
    # # Check if the returned text starts with 'Error:'
    # if mda_section_text.startswith("Error:"):
    #     print(f"    {mda_section_text}") # Print the specific error message
    # else:
    #     print(mda_section_text) # Print the extracted content
    # print("    --- End of Section ---")

  print(
      f"\nProcessing complete at {time.strftime('%Y-%m-%d %H:%M:%S')}."
      f' Processed {filings_processed_count} filings.'
  )


if __name__ == '__main__':
  # Ensure necessary libraries are installed: pip install requests beautifulsoup4 lxml
  main()
