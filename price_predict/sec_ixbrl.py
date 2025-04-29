# -*- coding: utf-8 -*-

from google3.ads.publisher.quality.micro_models.tensorflow.util import sec_common
from google3.ads.publisher.quality.micro_models.tensorflow.util.sec_common import large_mda_section_finder


# --- Main Execution ---
if __name__ == "__main__":
  target_viewer_url = "https://www.sec.gov/ix?doc=/Archives/edgar/data/1599407/000121390024100149/ea0220609-10q_1847hold.htm"
  target_viewer_url2 = "https://www.sec.gov/ix?doc=/Archives/edgar/data/1084869/000143774924032937/flws20240930_10q.htm"
  target_viewer_url3 = "https://www.sec.gov/ix?doc=/Archives/edgar/data/1770787/000177078724000073/txg-20240930.htm"
  target_viewer_url4 = "https://www.sec.gov/ix?doc=/Archives/edgar/data/1877461/000147793224007307/onesix_10q.htm"
  target_viewer_url5 = "https://www.sec.gov/ix?doc=/Archives/edgar/data/1600641/000160064124000194/dibs-20240930.htm"

  target_url = target_viewer_url5
  largest_mda_section = large_mda_section_finder(target_url)
  print("-" * 70)
  # Print the content of the largest section to standard output
  print(largest_mda_section)
  print("-" * 70)
  print(f"--- End of Largest MD&A Section passed in for {target_url}---")
