//! Output-path identity checks and same-directory atomic publication.

use std::{
    collections::BTreeSet,
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use crate::Error;

fn normalized_existing(path: &Path) -> Result<Option<PathBuf>, std::io::Error> {
    match fs::canonicalize(path) {
        Ok(normalized) => Ok(Some(normalized)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// Refuse output aliases to the source PDF or to another output.
///
/// # Errors
/// Returns an error for missing output directories or file aliases.
pub fn check_output_paths(input: &Path, outputs: &[&Path]) -> Result<(), Error> {
    let source = fs::canonicalize(input)?;
    let mut targets = BTreeSet::new();
    for output in outputs {
        let parent = output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let name = output
            .file_name()
            .ok_or_else(|| Error::Invalid("output must name a file".into()))?;
        let normalized = match normalized_existing(output)? {
            Some(path) => path,
            None => fs::canonicalize(parent)?.join(name),
        };
        if normalized == source {
            return Err(Error::Invalid(
                "output path aliases the original source PDF".into(),
            ));
        }
        if !targets.insert(normalized) {
            return Err(Error::Invalid(
                "two output paths resolve to the same file".into(),
            ));
        }
    }
    Ok(())
}

/// Write a completed output via a temporary file in its destination directory.
///
/// # Errors
/// Returns an error for failed file creation, write, flush, or persistence.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    {
        let mut writer = BufWriter::new(&mut temporary);
        writer.write_all(bytes)?;
        writer.flush()?;
    }
    temporary.persist(path).map_err(|error| error.error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{check_output_paths, write_atomic};
    use std::fs;

    #[test]
    fn outputs_cannot_alias_sources_or_each_other() -> Result<(), crate::Error> {
        let temp = tempfile::tempdir()?;
        let original = temp.path().join("source.pdf");
        let destination = temp.path().join("draft.json");
        fs::write(&original, b"sample")?;
        assert!(check_output_paths(&original, &[&original]).is_err());
        assert!(check_output_paths(&original, &[&destination, &destination]).is_err());
        check_output_paths(&original, &[&destination])?;
        write_atomic(&destination, b"one")?;
        write_atomic(&destination, b"two")?;
        assert_eq!(fs::read(destination)?, b"two");
        Ok(())
    }
}
