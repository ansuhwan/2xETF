from . import direxion, graniteshares, defiance, rex, tradr, leverage_shares

ALL_SCRAPERS = [
    ("Direxion", direxion.fetch),
    ("GraniteShares", graniteshares.fetch),
    ("Defiance", defiance.fetch),
    ("REX", rex.fetch),
    ("Tradr", tradr.fetch),
    ("LeverageShares", leverage_shares.fetch),
]
