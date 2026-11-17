# SPDX-License-Identifier: Apache-2.0
"""Checks that the repository is in English: fails if a tracked text file contains French.

    python tools/check_english.py            every text file of the repository
    python tools/check_english.py PATH...    only these files or folders

Two tests, line by line:
  1. a letter that English does not use (a, e, i, o, u with an accent, c with a cedilla, the oe ligature, guillemets);
  2. two or more words of a list of frequent French words that are not English words, or one word of a list of words
     that this project used in French (trame, treillis, sceau ...).

Allowed: the name of the project (Psi-treillis, psitreillis), what is listed in ALLOWED below, each entry with its
reason, and the part of CHANGELOG.md that follows the line HISTORY_MARKER: the entries written before this repository
was reorganised, which name files, options and messages as they were then. Binary files, the folders third_party/, build*/ and data/, and the bibliography with its verification notes (author names) are skipped.
Exit code 0 if nothing is found, 1 otherwise.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TEXT_SUFFIXES = {".md", ".txt", ".py", ".ps1", ".sh", ".cu", ".cuh", ".cpp", ".c", ".h", ".hpp", ".cmake", ".yml", ".yaml", ".json", ".csv",
                 ".log", ".cff", ".tex", ".def", ".toml", ".cfg", ""}
SKIPPED_FOLDERS = ("third_party/", "data/")
SKIPPED_FILES = {"paper/references.bib", "paper/references_verification.md", "paper/references_verification_addendum.md", "tools/check_english.py"}
HISTORY_FILE, HISTORY_MARKER = "CHANGELOG.md", "<!-- earlier history: names as they were -->"

ACCENTED = re.compile("[àâäçèéêëîïôöùûüœÀÂÇÈÉÊËÎÏÔÙÛŒ«»]")

# Frequent French words that are not English words (two on a line are needed: one alone can be a name or a typo).
COMMON = set("""le la les des du un une et ou est sont avec pour dans sur par pas que qui ce cette ces au aux en ne se sa son ses leur leurs nous
vous ils elle elles mais donc car si alors aussi ainsi comme tout tous toute toutes plus moins tres bien fait faire peut doit etre avoir ete
sans sous entre vers chez depuis pendant apres avant encore deja jamais toujours chaque autre autres meme memes dont ou quand lorsque
puis ensuite cela ceci celle celui ceux voici voila oui non""".split())
COMMON -= {"sur", "au", "son", "tout", "car", "non", "pour", "est", "ne", "ou", "se", "si", "sans", "comme", "fait", "dont", "plus", "bien", "avant", "en", "du", "des", "la", "le", "les", "un", "sa", "dans", "que", "pas", "par", "mais", "et", "ce", "aux"}
COMMON |= {"avec", "dans", "sont", "cette", "leur", "leurs", "nous", "vous", "ils", "elle", "elles", "donc", "alors", "aussi", "ainsi", "tous",
           "toute", "toutes", "moins", "peut", "doit", "etre", "avoir", "sous", "entre", "vers", "chez", "depuis", "pendant", "apres",
           "encore", "deja", "jamais", "toujours", "chaque", "autres", "meme", "memes", "quand", "lorsque", "puis", "ensuite", "cela", "ceci",
           "celle", "celui", "ceux", "voici", "oui", "une", "les", "des", "est", "pour", "par", "pas", "que", "mais"}

# Words this project used in French: one is enough.
PROJECT = set("""trame trames treillis sceau scelle empreinte empreintes sonde sondes ecretage ecrete echappement echappements debordement
debordements borne bornes taux ecart ecarts releve releves journal journaux banc bancs graine graines traceur traceurs noeud noeuds brique
briques niveau niveaux fichier fichiers lecteur lecture ecriture octet octets cle cles etat etats debut reglage reglages champ champs
revendication revendications tenue tenues mesure mesures calcule essai essais resultat resultats donnees sequence sequences chemin chemins
vitesse vitesses grille grilles pas_de_temps emetteur recepteur encodeur decodage encodage compression_taux erreur erreurs seuil seuils
plancher plafond refus garde verification verifier controle controles suite decoupe fenetre balayage validite durcissement resolution_des
comparateur figure_fr sortie sorties entree entrees ligne lignes colonne colonnes valeur valeurs moyenne ecart_type
nombre taille largeur hauteur attendu attendus trouve manque manquant manquants absent absente""".split())
PROJECT -= {"journal", "suite", "resolution_des", "sequence", "sequences", "figure_fr", "compression_taux", "pas_de_temps", "ecart_type",
            "verification", "garde", "borne", "refus", "absent", "essai", "plafond", "champ", "champs", "sortie", "entree", "mesure", "mesures",
            "tenue", "ligne", "valeur", "moyenne", "nombre", "taille", "niveau", "sonde", "trouve", "manque", "attendu", "calcule"}
PROJECT |= {"borne", "bornes", "refus", "essai", "essais", "champ", "champs", "sortie", "sorties", "entree", "entrees", "mesures", "ligne", "lignes",
            "valeur", "valeurs", "moyenne", "nombre", "taille", "niveau", "niveaux", "sonde", "sondes", "attendu", "attendus", "manque", "tenue",
            "tenues", "plafond", "garde"}

# Exact strings that are allowed, with the reason. Checked before the two tests: they are blanked out of the line.
ALLOWED = [
    "Psi-treillis", "psi-treillis", "psitreillis", "PsiTreillis", "PSITREILLIS", "Treillis",  # the name of the project
    "BORNE-", "BANC-", "LECT-1", "RELEC-1", "ROBU-1",                                            # former claim identifiers
    "Killian Pluenet",                                                                            # the author
    "Deriaz", "Perrier", "Lemarié-Rieusset", "Chacón", "Bürger",                   # names of cited authors
]

WORD = re.compile(r"[A-Za-z_][A-Za-z_]*")


def tracked_files(paths):
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "--"] + [str(p) for p in paths],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
        files = [ROOT / f for f in out if f]
    except (subprocess.CalledProcessError, FileNotFoundError):
        files = [p for base in paths for p in ([base] if Path(base).is_file() else Path(base).rglob("*")) if Path(p).is_file()]
    return sorted(set(files))


def findings_of(path):
    relative = path.relative_to(ROOT).as_posix()
    if relative.startswith(SKIPPED_FOLDERS) or relative in SKIPPED_FILES or path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    if any(part.startswith("build") for part in path.relative_to(ROOT).parts[:-1]):
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    found = []
    for number, line in enumerate(text.split("\n"), 1):
        if relative == HISTORY_FILE and line.strip() == HISTORY_MARKER:
            break
        probe = line
        for allowed in ALLOWED:
            probe = probe.replace(allowed, " ")
        accent = ACCENTED.search(probe)
        if accent:
            found.append((relative, number, f"letter {accent.group(0)!r}", line.strip()[:140]))
            continue
        words = [w.lower() for token in WORD.findall(probe) for w in token.split("_") if w]
        project = sorted({w for w in words if w in PROJECT})
        common = sorted({w for w in words if w in COMMON})
        if project:
            found.append((relative, number, "word " + ", ".join(project), line.strip()[:140]))
        elif len(common) >= 2:
            found.append((relative, number, "words " + ", ".join(common), line.strip()[:140]))
    return found


def main():
    paths = [Path(p).resolve() for p in sys.argv[1:]] or [ROOT]
    found = [f for path in tracked_files(paths) for f in findings_of(path)]
    for relative, number, what, line in found:
        print(f"{relative}:{number}: {what}: {line}")
    files = len({f[0] for f in found})
    print(f"{len(found)} line(s) in {files} file(s)" if found else "no French found")
    return 1 if found else 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    sys.exit(main())
