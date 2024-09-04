In total, close to **2.5 million images** have been indexed so far, with new datasets added regularly.

**Rijksmuseum**

- Source: [https://data.rijksmuseum.nl/object-metadata/download/](https://data.rijksmuseum.nl/object-metadata/download/)
- Size: **392636**
- Method: Public CSV
- License: [CC0](https://creativecommons.org/publicdomain/zero/1.0/), see [https://www.rijksmuseum.nl/en/research/conduct-research/data/policy](https://www.rijksmuseum.nl/en/research/conduct-research/data/policy)
- Year added: 2020
- Completeness: 100% (no errors at time of scraping)
- Metadata: backlink to website
- Type: remote (images pulled from server on demand)
- Embeddings: VGG19, Poses, Raw, CLIP

**Museum of Modern Art (MoMA)**

- Source: [https://www.moma.org/collection/](https://www.moma.org/collection/)
- Size: **72365**
- Method: Brute-force scraping (now restricted)
- License: publicly accessible, no download
- Year added: 2020
- Completeness: ca. 75% (good approximation of collection) – some images blocked by missing CORS headers which means they cannot be "translated" into other datasets by extracting embeddings
- Metadata: backlink to website
- Type: remote (images pulled from server on demand)
- Embeddings: VGG19, Poses, Raw, CLIP

**Metropolitan Museum (Met)**

- Source: [https://www.kaggle.com/metmuseum/the-met](https://www.kaggle.com/metmuseum/the-met)
- Size: **358426**
- License: [CC0](https://creativecommons.org/publicdomain/zero/1.0/), see [https://www.metmuseum.org/about-the-met/policies-and-documents/open-access](https://www.metmuseum.org/about-the-met/policies-and-documents/open-access)
- Method: BigQuery, see [https://cloud.google.com/bigquery/docs/quickstarts/quickstart-cloud-console](https://cloud.google.com/bigquery/docs/quickstarts/quickstart-cloud-console)
- Details:
    - `SELECT bigquery-public-data.the_met.images.original_image_url, bigquery-public-data.the_met.objects.link_resource`
    - `FROM bigquery-public-data.the_met.objects JOIN bigquery-public-data.the_met.images ON bigquery-public-data.the_met.objects.object_id = bigquery-public-data.the_met.images.object_id`
- Year added: 2021
- Completeness: 99% (no errors at time of scraping, some URLs dead now)
- Metadata: backlink to website
- Type: remote (images pulled from server on demand)
- Embeddings: VGG19, Poses, Raw, CLIP

**Smithsonian**

- Source: [https://github.com/Smithsonian/OpenAccess](https://github.com/Smithsonian/OpenAccess)
- Size: **136455**
- License: [CC0](https://creativecommons.org/publicdomain/zero/1.0/), see [https://www.si.edu/openaccess/faq](https://www.si.edu/openaccess/faq)
- Method: Public JSON
- Year added: 2022
- Completeness: 99% (no errors at time of scraping, some files moved in the meantime)
- Metadata: backlink to website
- Type: remote (images pulled from server on demand)
- Embeddings: VGG19, Poses, Raw, CLIP

**Smithsonian_Local**

- Local mirror of the Smithsonian dataset

**ImageNet**

- Source: [https://www.kaggle.com/c/imagenet-object-localization-challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge) – this is the ILSVRC 2012 version!
- Size: **1322677**
- License: publicly accessible
- Method: Kaggle download
- Details: `kaggle competitions download -c imagenet-object-localization-challenge`
- Year added: 2024
- Completeness: 92% (8% non-readable images)
- Metadata: none
- Type: (hosted on imgs.ai)
- Embeddings: CLIP

**Getty**

- Source: [https://www.getty.edu/projects/open-content-program/](https://www.getty.edu/projects/open-content-program/)
- Size: **84256**
- License: [CC0](https://creativecommons.org/publicdomain/zero/1.0/), see [https://www.getty.edu/projects/open-content-program/](https://www.getty.edu/projects/open-content-program/)
- Method: unknown
- Year added: 2024
- Completeness: unknown
- Metadata: none
- Type: local (hosted on imgs.ai)
- Embeddings: VGG19, Poses, Raw, CLIP